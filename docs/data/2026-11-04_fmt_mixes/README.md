# Q219, Q889, Q901, Q888: four format levers for Swift Qwen3.8-Flash-Next (2026-11-04)

[pascal-mix](../2026-11-04_pascal_mix/README.md) showed that choosing each tensor's weight type for the P106 pays:
`pmix-q2-fast` keeps the makers' routed experts and gives the dense weights the P106's fastest types, at Swift Q2_0's
quality (agent KLD 0.279 against 0.287) and est. 10 ms a token less. This folder codes four more levers on top of it,
each with pascal-mix's tools, and builds the files that price them:

1. **Q219, the routed experts allocated per 8-layer group** across the published GSQ builds, with
   [Q167](../2026-10-27_ornith_alloc/README.md)'s method (on Ornith-1.5-122B the allocation mattered more than the
   format itself): one switch at a time, each scored for KLD, the costs added, the best set within a byte budget.
2. **Q889 (C326), a spine by sensitivity:** `pmix-q2-fast` with one dense tensor class at a time raised to Q6_K (the
   output layer to Q8_0), so that each class's KLD price is known.
3. **Q901 (C339), a verify-shaped mix:** each dense tensor's type chosen to minimise the expected time of a verify
   pass, Σ_k P(k) · T(type, k), under pmix-q2-fast's bytes and KLD.
4. **Q888 (C323), the dense types the selector lacks timings for**, at Flash-Next's shapes on one P106: the rig's run
   file.

**Status:** the tools are written and tested; all 30 single-switch files (8 for C326, 22 for Q219) and the control
`dniq23` are built and scored on all three texts: agent ([results](#results-so-far-agent-text-kld-2026-11-05)), wiki
([results](#wiki-text-kld-2026-11-06)) and code ([results](#code-text-kld-2026-11-10), with the three texts side by
side); the allocation over the three texts (`galloc.py solve`) is solved and its pick at the start's bytes, `gmix-q2`,
is built ([the v3.1 pick](#the-v31-pick-2026-11-11)); its measured KLD (Q1366) and its rig check (Q1367) come next; the
rig timing (C323) is a run file for a rig. Registry rows Q219, Q888, Q889, Q901, Q1366, Q1367 in
[quants/experiments.md](../../../quants/experiments.md).

## Results so far: agent-text KLD (2026-11-05)

Wiki text and code text follow in their own sections ([wiki](#wiki-text-kld-2026-11-06),
[code](#code-text-kld-2026-11-10)). Each of
the 30 single-switch files and the control `dniq23` was scored against the Swift-1.5 Q8_0's logits on the agent text, on the same five cards as
`pmix-q2-fast` and spine6 ([kld_agent.tsv](kld_agent.tsv): KLD and its ±, the change from the start, the same-top-token
share and each file's class). A **gain** is below `pmix-q2-fast`'s 0.2792 by more than its ± (0.0054,
`llama-perplexity`'s standard error of the mean KLD), a **loss** above it by more, the rest **negligible**. All the files
share the text and the reference logits, so their differences are tighter than the ±: the classes are conservative.

- **18 gains, 12 negligible, no loss.**
- **C326's arms** (one dense class at spine6's types): gains for the attention output (−0.0098 for +25 MB, est.
  +0.3 ms: the best per byte and per ms), the shared expert (−0.0083), recurrent `attn_qkv` (−0.0066) and attention
  `q`/`k`/`v` (−0.0064); negligible for `ssm_out` (−0.0037), `blk.1.ple_key` (−0.0039), the Q8_0 output layer
  (−0.0005) and `attn_gate` (+0.0008). The four gains add to −0.031, more than spine6's whole −0.0233, so they overlap.
- **Q219's switches** (one class of one 8-layer group from another GSQ build): the best are the dense tensors of groups 1
  and 0 from `pmix-iq3-fast` (`denseiq31` 0.2599, −0.0193 for +56 MiB; `denseiq30` 0.2629), Swift IQ2_XS's gate/up in
  groups 0 and 3 (`guiq20` 0.2621, `guiq23` 0.2631) and Swift IQ3_XXS's in groups 2 and 3 (`guiq32` 0.2640, `guiq33`
  0.2643). IQ3_XXS's down projections pay in group 3 only (`dniq33` −0.0103); in group 5 they add 1.1 GB for no better
  KLD (`dniq35` 0.2840, +0.0048, the worst file). The same-top-token share stays within 0.4 points (90.70 % at the
  start, 91.04 % at most).
- **Read with care:** IQ2_XS's gate/up, at IQ3_XXS's bytes or fewer (`guiq21` is 0.24 GB smaller than the start), gains
  as much as IQ3_XXS's in the same groups, and `pmix-iq3-fast`'s dense tensors gain in groups 0–2 and nothing in groups
  3–5. Part of each gain may come from mixing two builds' rounding rather than from the types: the control below
  prices that.
- **The control `dniq23`** (IQ2_XS's own Q2_0 down projections of layers 24–31, the start's type, no bytes added):
  agent KLD **0.275440 ± 0.005310** against the start's 0.2792, **−0.0038**, same top token 90.751 ± 0.131 % (90.70 % at
  the start): negligible. Mixing builds alone costs no quality; it moves the score by −0.0038, in the direction of a
  gain and within the ±. Read the 14 gains of Q219's switches against it: nine are more than 0.0054 below the control
  (`guiq33`, `guiq32`, `dniq33`, `denseiq32`, `denseiq31`, `denseiq30`, `guiq23`, `guiq22`, `guiq20`; `dniq33`, IQ3_XXS's
  down projections of the control's layers, is 0.0065 below it, so its IQ4_NL type pays beyond the mixing), and five are
  not: `guiq35` (−0.0088), `guiq34` (−0.0055), `guiq30` (−0.0073), `guiq25` (−0.0068) and `guiq21` (−0.0083) are less
  than 0.0054 below it. One control covers one class in one group (down projections of layers 24–31), so it is no
  proof for the gate/up and dense switches.
- **The allocation** over all three texts and its first pick: [the v3.1 pick](#the-v31-pick-2026-11-11).

## Wiki-text KLD (2026-11-06)

The same 31 files on wikitext-2 test (80 × 512 tokens) against the Swift-1.5 Q8_0's wiki logits, on the same five
cards ([kld_wiki.tsv](kld_wiki.tsv): the agent table's columns, then the change from the control and the class against
it; written from the logs by [run/kld_table.py](run/kld_table.py), which rebuilds `kld_agent.tsv` exactly from the
agent logs). The start `pmix-q2-fast` scores **0.426376 ± 0.005852** (same top token 77.706 ± 0.291 %), spine6 0.4211
(−0.0053). The classes are the agent text's: a **gain** is below the start by more than its ± (0.0059), a **loss**
above it by more, the rest **negligible**; against the control, the same with the control's ± (0.0059).

- **13 gains, 17 negligible, no loss** among the 30 single switches.
- **C326's arms gain nothing on wiki text.** All eight are within the ±: the shared expert −0.0056, recurrent
  `attn_qkv` −0.0037, `ssm_out` −0.0033, the Q8_0 output layer −0.0017, `blk.1.ple_key` −0.0005, attention
  `q`/`k`/`v` −0.0001, the attention output +0.0002 (the agent text's best per byte) and `attn_gate` +0.0027. The whole
  spine6 (−0.0053) is within it too.
- **Q219's routed experts of the late groups gain most.** Group 5's gate/up (layers 40–47) from IQ3_XXS, `guiq35`:
  0.3833, −0.0431, same top token 78.85 % (+1.14 points); from IQ2_XS, `guiq25`: 0.3942, −0.0322 for +500 MiB, under
  half IQ3_XXS's +1,075 MiB. Group 4's gate/up: `guiq34` −0.0265, `guiq24` −0.0158. IQ3_XXS's down projections of
  groups 4 and 5: `dniq34` −0.0254 and `dniq35` −0.0169 (the worst file on agent text, +0.0048). Groups 0–3 gain less
  (−0.0064 to −0.0145), and group 1's gate/up gains nothing from either build (`guiq31` −0.0046, `guiq21` −0.0007).
  The dense switches, three of them among the agent text's best (`denseiq31` −0.0193 there), are within the ± on wiki
  (−0.0022 to +0.0007).
- **The control `dniq23`:** **0.429660 ± 0.005919**, +0.0033 against the start, same top token 77.637 ± 0.292 %:
  negligible, and on the other side of the start from its agent score (−0.0038). Mixing builds alone moves neither
  text beyond the ±, and each of the 13 wiki gains is more than 0.0059 below the control too. Against the control 18
  switches gain: those 13 and five that clear it only because the control sits above the start (`s6shexp`,
  `s6gdn_qkv`, `s6gdn_out`, `guiq31`, `dniq30`: −0.0066 to −0.0089 from the control, within the ± from the start).
- **The two texts together:** 10 switches gain on both (`guiq35`, `guiq34`, `guiq33`, `guiq32`, `guiq30`, `guiq25`,
  `guiq23`, `guiq22`, `guiq20`: every gate/up switch but group 1's two and `guiq24`; and `dniq33`). Eight gain on agent
  text only (the four arms that gained there, the dense switches of groups 0–2, `guiq21`), three on wiki only
  (`dniq34`, `dniq35`, `guiq24`), and none loses on either. Agent text, the one the agent check reads, rewards the
  dense tensors and the early groups; wiki text rewards the late groups' routed experts.

## Code-text KLD (2026-11-10)

The same 31 files on the agent check's Exercism code (80 × 512 tokens) against the Swift-1.5 Q8_0's code logits
([kld_code.tsv](kld_code.tsv): the wiki table's columns, written by [run/kld_table.py](run/kld_table.py)). The 31
files ran on four cards (two RTX 3080s, two RTX 3090s), and the start is scored on the same four: `pmix-q2-fast`
scores **0.108328 ± 0.003850** there (same top token 96.059 ± 0.136 %), spine6 0.104199 (−0.0041, past the ± by
0.0003). The classes are the other texts': a **gain** is below the start by more than its ± (0.0039), a **loss** above
it by more, the rest **negligible**; against the control, the same with the control's ± (0.0039).

- **9 gains, 21 negligible, no loss** among the 30 single switches.
- **C326's arms gain nothing on code text.** All eight are within the ±: `ssm_out` −0.0021, recurrent `attn_qkv`
  −0.0012, `attn_gate` −0.0010, the shared expert −0.0006, the Q8_0 output layer −0.0003, the attention output −0.0003,
  attention `q`/`k`/`v` −0.0002 and `blk.1.ple_key` −0.0002.
- **Every gain is a routed-expert switch of Q219, eight of the nine a gate/up switch.** Group 0 (layers 0–7): `guiq30`
  0.099531 (−0.0088, with 100 MiB fewer than the start) and `guiq20` 0.100677 (−0.0077). Group 5 (layers 40–47):
  `guiq35` 0.100162 (−0.0082) and `guiq25` 0.101137 (−0.0072, +500 MiB against IQ3_XXS's +1,075). Then `guiq34`
  0.103414 (−0.0049), IQ3_XXS's down projections of group 4, `dniq34` 0.103701 (−0.0046), `guiq32` 0.103754 (−0.0046),
  `guiq22` 0.103848 (−0.0045) and `guiq31` 0.104421 (−0.0039, past the ± by 0.00006). Within the ±: `dniq33` −0.0033,
  `guiq21` −0.0033, `guiq24` −0.0031, `dniq30` −0.0029, `dniq35` −0.0029, `guiq23` −0.0020, `guiq33` −0.0019. The six
  dense switches move nothing: `denseiq31` −0.0012, `denseiq32` −0.0009, `denseiq35` −0.0006, `denseiq30` −0.0003,
  `denseiq34` −0.0001, `denseiq33` +0.0007. The same-top-token share runs from 96.02 % (`s6gdn_gate`) to 96.27 %
  (`guiq35`), 96.06 % at the start.
- **The control `dniq23`:** **0.109187 ± 0.003860**, +0.0009 against the start, same top token 96.098 ± 0.136 %:
  negligible, as on the other two texts (agent −0.0038, wiki +0.0033). Each of the nine gains is more than 0.0039
  below the control too. Against the control 12 switches gain: those nine and `dniq33`, `guiq21` and `guiq24`, which
  clear it only because the control sits above the start.
- **The start on both card sets (Q1185).** Every agent and wiki score here is from five cards; the code text ran on
  four. With the same `llama-perplexity`, the same file and the same `-fit on` split, `pmix-q2-fast` scores **0.107982
  ± 0.003843** on code text on the five cards (same top token 96.118 ± 0.135 %) and **0.108328 ± 0.003850** on the
  four, 0.0003 higher. The table takes the four-card score, so every change in it compares two runs on the same cards.
  One class depends on which score is the start: `guiq31` passes the ± by 0.00006 against the four-card score and
  falls 0.0003 short of it against the five-card one (−0.0036). The other 29 switches keep their class either way;
  the nearest, `guiq22`, `guiq32` and `dniq34`, pass the ± by 0.0006, 0.0007 and 0.0008 (0.0003, 0.0004 and 0.0004
  against the five-card score). Each score is one run: how far a repeat on the same cards moves it is not measured
  (Q1194).
- **Each file ran whole on the four cards:** the cards' memory peaks sum to at least 2,815 MiB more than the file's two
  weight shards ([mem_peak_code.tsv](mem_peak_code.tsv); the start's peaks sum to 38,824 MiB, 2,818 more than its
  shards). The start on these four cards with another build of `llama-perplexity` scored 0.107518 on code text
  ([N2P files](../2026-11-04_n2p_flash/README.md), `kld.tsv`).

### The three texts together

[kld_texts.tsv](kld_texts.tsv) ([run/kld_texts.sh](run/kld_texts.sh)) puts each file's three changes and classes side
by side.

- **Seven switches gain on all three texts**, all of them routed gate/up: Swift IQ3_XXS's in groups 5, 4, 2 and 0
  (`guiq35`, `guiq34`, `guiq32`, `guiq30`) and Swift IQ2_XS's in groups 5, 2 and 0 (`guiq25`, `guiq22`, `guiq20`).
  `guiq30` does it with 100 MiB fewer than the start (agent −0.0073, wiki −0.0066, code −0.0088).
- **Four gain on two texts:** group 3's three routed switches on agent and wiki (`guiq33`, `guiq23`, `dniq33`), and
  `dniq34` on wiki and code.
- **Eleven gain on one text:** on agent text only, the four arms that gained there (`s6fa_out`, `s6shexp`,
  `s6gdn_qkv`, `s6fa_qkv`), the dense switches of groups 0–2 (`denseiq30`, `denseiq31`, `denseiq32`) and `guiq21`; on
  wiki text only, `dniq35` and `guiq24`; on code text only, `guiq31`, by 0.00006.
- **Eight gain on none:** `s6gdn_gate`, `s6gdn_out`, `s6ple`, `q8output`, `dniq30`, `denseiq33`, `denseiq34`,
  `denseiq35`.
- **No switch loses on any text.** The largest rise is `dniq35`'s +0.0048 on agent text, within that text's ±
  (0.0054).
- **What the texts agree on:** better routed gate/up pays on every text in groups 0, 2 and 5 from either build and in
  group 4 from IQ3_XXS. Group 3's pays on agent and wiki text. Group 1's pays on one text per build: agent text from
  IQ2_XS (`guiq21`), code text from IQ3_XXS (`guiq31`, by 0.00006). The dense classes, the agent text's best per byte
  (`denseiq31`, the attention output), pay on agent text only.
- **The allocation over the three texts** follows in [the v3.1 pick](#the-v31-pick-2026-11-11). `galloc.py collect`
  read each switch's change from the start's score on the same text from the logs: for code text the start's
  four-card log (the log folder of the [commands](#commands) in which it stands under the start's name).

## The v3.1 pick (2026-11-11)

`galloc.py solve` over the three texts ([solve.tsv](solve.tsv); each switch's cost in
[solve-switches.tsv](solve-switches.tsv)), with Q219's switches of [switches.tsv](switches.tsv) and the scores of the
three tables (`kld.tsv` from `galloc.py collect`). The damage is Q167's, the sum over the texts of each switch's KLD
change relative to the start's KLD on that text, so a fraction of the start counts the same on every text. Three
budgets on the cards: `pmix-q2-fast`'s own bytes (37,470,389,760, so the pick fits wherever the start fits), and the
44.3 GB and 47 GB of [the allocation](#1-q219-the-routed-experts-per-8-layer-group). C326's arms are not options: they
take tensors that the dense switches take too, and none of them gains on wiki or code text.

| Budget on the cards | Switches | Bytes added | Est. ms a token | Predicted KLD agent / wiki / code | ± of the summed switches |
|---|---|---:|---:|---|---|
| `pmix-q2-fast` (start, measured) | — | — | 72.44 | 0.279228 / 0.426376 / 0.108328 | — |
| 37.47 GB, the start's | `guiq30`, `guiq21`, `denseiq31`, `denseiq32`, `guiq23` | −25,891,840 | 78.35 | **0.21610 / 0.41379 / 0.09212** | 0.01148 / 0.01304 / 0.00840 |
| 44.3 GB | 15: gate/up `guiq20`, `guiq21`, `guiq22`, `guiq23`, `guiq24`, `guiq35`; down `dniq30`, `dniq33`, `dniq34`, `dniq35`; dense of groups 0, 1, 2, 4, 5 | +6,643,681,280 | 88.80 | 0.14835 / 0.28020 / 0.06266 | 0.02028 / 0.02232 / 0.01464 |
| 47 GB | 15: gate/up `guiq20`, `guiq31`, `guiq32`, `guiq23`, `guiq34`, `guiq35`; down and dense as at 44.3 GB | +9,068,513,280 | 91.19 | 0.14933 / 0.26671 / 0.06007 | 0.02030 / 0.02230 / 0.01464 |

The est. ms are the timed kernels of `bench/file_choice/efficiency.tsv`, in which IQ2_XXS and IQ2_XS gate/up run at
llama.cpp's IQ2_S speed (27.5 GB/s; C2's Pascal kernel is expected at 42.6), so the IQ2_XS switches cost less than
shown once that kernel lands. The ± column is the root sum of squares of the picked switches' own ±. All the files
share the text and the reference logits, so the ± of a difference between them is smaller.

**The pick at the start's bytes is `gmix-q2`,** built as one file (below). Group by group:

| Group (layers) | Switch | What changes | MiB on the cards | Change agent / wiki / code |
|---|---|---|---:|---|
| 0 (0–7) | `guiq30` | routed gate/up from Swift IQ3_XXS (its IQ2_XS ×8, IQ2_XXS ×8 in these layers) | −100.0 | −0.00727 / −0.00661 / −0.00880 |
| 1 (8–15) | `guiq21` | routed gate/up from Swift IQ2_XS (IQ2_XXS ×8, IQ1_M ×4, IQ2_S ×4) | −225.0 | −0.00828 / −0.00067 / −0.00327 |
| 1 (8–15) | `denseiq31` | the other tensors from `pmix-iq3-fast` | +55.7 | −0.01933 / +0.00072 / −0.00125 |
| 2 (16–23) | `denseiq32` | the other tensors from `pmix-iq3-fast` | +44.6 | −0.01208 / +0.00033 / −0.00088 |
| 3 (24–31) | `guiq23` | routed gate/up from Swift IQ2_XS (IQ2_S ×10, IQ2_XXS ×6) | +200.0 | −0.01617 / −0.00636 / −0.00201 |
| groups 4 and 5 | — | the start's | 0 | — |
| **sum** | | | **−24.7** | **−0.06313 / −0.01259 / −0.01621** |

- **What it predicts:** agent KLD 0.279228 → 0.21610 (−0.06313), wiki 0.426376 → 0.41379 (−0.01259), code 0.108328 →
  0.09212 (−0.01621), at 24.7 MiB fewer on the cards. Agent text gains most: three of the five switches gain on agent
  text only (`denseiq31`, `denseiq32`, `guiq21`), and `guiq21` is picked mostly for its 225 MiB, which pay for the
  others.
- **What it costs:** est. +5.91 ms a token on the timed kernels (78.35 against 72.44), +4.21 ms on the expected ones;
  4.68 ms of the 5.91 are the three gate/up switches, whose IQ1_M, IQ2_XXS, IQ2_XS and IQ2_S tensors run at IQ2_S's
  speed in that table.
- **The 44.3 GB and 47 GB picks** take every group's routed experts from a better build and the dense tensors of five
  groups; they are solved, not built ([Q1196](../../../quants/experiments.md)'s size ladder builds files at these sizes
  with more donors).

**Do the switches add up?** The prediction sums single switches. `pmix-iq3-fast` is the start with every IQ3_XXS
switch at once (16 scored switches, plus the Q2_0 down projections of groups 1–2 in IQ3_XXS's rounding, which no
switch file holds, and its token embeddings, which stay in host memory, in another type), and it is measured
([pascal-mix](../2026-11-04_pascal_mix/README.md)'s `kld.tsv`, five cards). Its change from the start against the sum
of the 16 switches' changes ([additivity.tsv](additivity.tsv), [run/additivity.py](run/additivity.py)):

| Text | Sum of the 16 switches | `pmix-iq3-fast`, measured | Whole over sum |
|---|---:|---:|---:|
| agent | −0.11758 | −0.10572 | 0.899 |
| wiki | −0.16071 | −0.17732 | 1.103 |
| code | −0.04866 | −0.05289 | 1.087 |

So over a whole build the single switches' changes add up to within 11 % of the measured change, less than the sum on
agent text and more on wiki and code. Q167 accepted a prediction within two standard errors of the measured KLD; the
measurement of `gmix-q2` on the three texts is the test (Q1366).

**The file.** `run/build_pick.sh` with `galloc.py final`'s two steps, on four CPU threads at nice 19 and idle I/O
priority, 17 min 6 s on 2026-11-11: step 1 takes 372 tensors from `pmix-iq3-fast` (`guiq30`, `denseiq31`,
`denseiq32`: +322,560 bytes), step 2 takes 32 from Swift IQ2_XS (`guiq21`, `guiq23`: −26,214,400 bytes), and
`gguf_compose.py verify` passes both: every one of the 1,224 tensors is byte for byte its donor's or the step's base.
Shards 1 and 2 hold 10,787,009,120 and 26,941,664,896 bytes (md5 `35f5b9f67c343f55969f91cfd5835c98` and
`e36684c6ee4724b06a24322cf54efca8`), 25,891,840 fewer than `pmix-q2-fast`'s, as the solve said; shard 3 is the
start's n-gram table, unchanged. Every tensor type in it is one that `pmix-iq3-fast` or Swift IQ2_XS already holds.

**Its fit at 64k on nine 6 GB cards**, from the headers ([run/pick_fit.sh](run/pick_fit.sh)): `split_plan.py` on the
two nine-card lines whose costs [fewer_hops](../2026-11-04_fewer_hops/README.md) calibrated against llama.cpp's
projection on the rig (f16 cache, `-b 2048 -ub 512`), both files on the same split; the start's rows match the rig's
projection to 1 MiB a card ([pick_fit_64k_cards.txt](pick_fit_64k_cards.txt)). MiB on each card, `pmix-q2-fast` →
`gmix-q2`:

| Line | Card 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | Fullest card's free MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no draft, `-ts 6,5,5,6,5,5,6,5,6` | 4,943 → 4,918 | 4,243 → 4,029 | 4,427 → 4,396 | 5,001 → 5,033 | 4,271 → 4,471 | 4,432 → 4,445 | 5,009 | 4,276 | 4,868 | 1,014 → 990 |
| MTP draft's body on card 8, `-ts 6,6,6,6,5,6,5,6,3` | 4,990 → 4,965 | 5,179 → 4,929 | 5,047 → 5,070 | 5,200 → 5,227 | 4,312 → 4,524 | 5,057 → 5,045 | 4,461 | 5,064 | 4,632 | 823 → 796 |

Card 4 gains 200 to 212 MiB and card 1 loses 214 to 250; cards 6–8 hold only unswitched layers. The fullest card is
card 3, with 24 and 27 MiB less free than the start's fullest (card 6 without a draft, card 3 with it). With the MTP
line's 645 MiB of draft token embeddings on card 4 (`-otd token_embd=CUDA4`) that card keeps 854 MiB free, more than
card 3. `fit.py`'s whole grid ([fit-gmix-q2.tsv](fit-gmix-q2.tsv), beside
[pmix-q2-fast's](../2026-11-04_n2p_flash/fit/fit-pmix-q2-fast.tsv), which it reproduces byte for byte) gives every 64k
line of the start a whole-layer split for `gmix-q2`: the same free MiB on the fullest card without a draft at every
micro-batch, 9 to 21 MiB less with the MTP draft, 34 to 109 MiB more with DFlash2.

## 1. Q219: the routed experts per 8-layer group

Tool: [bench/pascal_mix/galloc.py](../../../bench/pascal_mix/galloc.py) (`specs`, `switches`, `same`, `collect`,
`solve`, `final`), with its self-test [galloc_test.py](../../../bench/pascal_mix/galloc_test.py).

- **The start** is `pmix-q2-fast`: 37.47 GB on the cards (everything but the token embeddings and the n-gram table),
  est. 72.4 ms a decoded token on `-ladder3` ([bench/file_choice](../../../bench/file_choice/README.md)'s model).
- **A switch** moves one class of one 8-layer group (layers 0–7, …, 40–47) to a donor file's tensors, byte for byte
  ([scripts/gguf_compose.py](../../../scripts/gguf_compose.py), split-aware: each file keeps the start's three shards, a
  shard with nothing taken is a link):
  - `guiq3<g>`: the routed experts' gate and up from Swift IQ3_XXS (taken from `pmix-iq3-fast`, which holds them byte
    for byte);
  - `dniq3<g>`: their down projection from Swift IQ3_XXS;
  - `guiq2<g>`: gate and up from Swift IQ2_XS;
  - `denseiq3<g>`: every other tensor of the group's layers (attention, recurrent, shared expert, router, norms) from
    `pmix-iq3-fast`.
- **Which switches change anything:** in the GSQ builds each layer has its own expert type (Swift IQ3_XXS's gate/up run
  from IQ2_XXS to IQ3_XXS by layer, its down is IQ4_NL in 18 layers and Q2_0 in the rest; Swift IQ2_XS mixes IQ1_M,
  IQ2_XXS and IQ2_S in gate/up and keeps Q2_0 down everywhere). The down projections of groups 1 and 2 of IQ3_XXS, and
  all of IQ2_XS's, have the start's type (Q2_0). Their bytes still differ: **all 48 Q2_0 down projections of IQ2_XS
  and of IQ3_XXS differ from Swift Q2_0's** (`galloc.py same`), so the builds don't share GSQ's rounding (each block was
  calibrated on its own build's earlier blocks). One such switch is built as a control, `dniq23` (IQ2_XS's own Q2_0
  down of layers 24–31): its KLD is the cost of mixing builds alone.
- **The 23 files** ([switches.tsv](switches.tsv): tensors, type changes, bytes and est. ms each switch adds):

  | Group (layers) | 0 (0–7) | 1 (8–15) | 2 (16–23) | 3 (24–31) | 4 (32–39) | 5 (40–47) |
  |---|---|---|---|---|---|---|
  | gate/up from IQ3_XXS: GB, est. ms | −0.10, +1.8 | +0.34, +1.8 | +1.10, +2.3 | +0.81, +2.1 | +1.42, +2.3 | +1.13, +1.6 |
  | down from IQ3_XXS (its IQ4_NL layers) | +0.71, +0.8 (3) | — | — | +0.94, +1.0 (4) | +1.42, +1.6 (6) | +1.18, +1.3 (5) |
  | gate/up from IQ2_XS | +0.31, +1.3 | −0.24, +1.5 | +0.31, +1.3 | +0.21, +1.4 | +0.35, +1.2 | +0.52, +1.2 |
  | other tensors from pmix-iq3-fast | +0.09, +0.9 | +0.06, +0.7 | +0.05, +0.6 | +0.07, +0.9 | +0.06, +0.7 | +0.05, +0.6 |

  All the IQ3_XXS switches together are `pmix-iq3-fast` (46.80 GB, est. 93.4 ms). The est. ms are the timed kernels of
  `bench/file_choice/efficiency.tsv`; IQ2_XXS and IQ2_XS gate/up are there at llama.cpp's IQ2_S speed (27.5 GB/s; C2's
  Pascal kernel is expected at 42.6), so their switches are dearer in ms than they will be.
- **The allocation** (`galloc.py solve`): each switch's damage is Q167's, the sum over the three texts of its KLD
  change relative to the start's KLD on that text. A group's options exclude each other within a class (gate/up from
  IQ3_XXS or IQ2_XS, not both). The best set at each budget is found exactly: every combination of each group's options
  (a Pareto front of bytes, damage and est. ms, merged group by group), so an ms budget (`--ms-budget`) can be set
  beside the bytes. The budgets are 44.3 GB on the cards (the ceiling with the MTP draft's ~2.7 GB beside the model,
  [sub_1_bit_quants.md](../../research/notes/sub_1_bit_quants.md) §1) and 47 GB (no draft). The self-test checks the
  solver against brute force over three groups' options on a synthetic score table.
- **Then** the picks are built as one file each (`run/build_pick.sh`: one compose per donor in turn, each verified)
  and scored like the switches; Q167 accepted a prediction within two standard errors of the measured KLD.

## 2. Q889 (C326): a spine by sensitivity

pmix-q2-spine6 (every dense matrix at Q6_K or above) cut pmix-q2-fast's agent KLD by 0.023 for est. +9.6 ms a token.
Which classes pay? Each arm is pmix-q2-fast with one class taken from pmix-q2-spine6 (Q6_K where the start was below
it; the shared expert's 640-wide down at Q5_0), or from the Swift-1.5 Q8_0 for the output layer: pure composes, no
requantization ([spine_arms.tsv](run/spine_arms.tsv), [spine_switches.tsv](spine_switches.tsv)):

| Arm | Class (tensors) | Changes | MB on the cards | Est. ms a token |
|---|---|---|---:|---:|
| `s6gdn_qkv` | recurrent layers' `attn_qkv` (36) | IQ4_XS ×31, Q4_K ×4, Q2_0 ×1 → Q6_K | +276.1 | +3.07 |
| `s6gdn_gate` | recurrent layers' `attn_gate` (36) | IQ4_XS ×29, Q2_0 ×4, Q4_K ×3 → Q6_K | +177.9 | +1.99 |
| `s6gdn_out` | `ssm_out` (36) | IQ4_XS ×20, Q4_K ×11, Q5_K ×3 → Q6_K | +141.8 | +1.60 |
| `s6fa_qkv` | attention layers' `attn_q`, `attn_k`, `attn_v` (36) | 32 → Q6_K | +131.2 | +1.47 |
| `s6shexp` | the shared expert's gate, up, down (144) | 124: gate/up → Q6_K, down → Q5_0 | +59.9 | +0.96 |
| `s6fa_out` | `attn_output` (12) | Q4_K ×4, Q5_K ×4 → Q6_K | +24.6 | +0.28 |
| `s6ple` | `blk.1.ple_key` (1) | Q2_0 → Q6_K | +14.1 | +0.17 |
| `q8output` | the output layer | Q5_K → Q8_0 | +238.4 | +20.1 timed (+4.6 expected) |

The seven Q6_K arms sum to est. +9.5 ms, spine6's +9.6. The hyper-connections have no arm: the makers' file keeps them
in BF16, the model's own precision, which Q6_K or Q8_0 would only lower. The arms' KLD prices each class
(ΔKLD per ms and per MB), and `vmix.py calib` turns them into the per-class gains of the selector below.

## 3. Q901 (C339): the verify-shaped selector

Tool: [bench/pascal_mix/vmix.py](../../../bench/pascal_mix/vmix.py) (`times`, `pk`, `calib`, `select`), with the
error measurement [bench/pascal_mix/qerr.py](../../../bench/pascal_mix/qerr.py).

**P(k).** S33's MTP draft line (`mtp2d`, n-max 3) on the agent check, from llama-server's counters over the whole arm
(`vmix.py pk`): 4,509 draft passes, each of 3 drafted tokens (so every verify is a k = 4 pass), acceptance 0.723, and
51 tokens decoded outside them: **P(4) = 0.989, P(1) = 0.011**. With a fixed draft length the pass size does not depend
on the acceptance; an adaptive length (C134) would spread it, and `vmix.py pk` reads the counters of any arm.

**T(type, shape, k).** `vmix.py times` merges the P106 tables into one ([times_kshape.tsv](times_kshape.tsv), 256
rows): test-backend-ops perf inside ggml ([k-curve](../2026-11-04_kcurve_p106/README.md)'s `perf_dense.tsv` and
`perf_cov.tsv`; with `--dispatch min` the faster of the k-column and the other kernels, as `-kshape`'s dispatch table
intends), [kshape](../2026-11-04_kshape/README.md)'s `ggml_est.tsv`, and the bench's every launch shape
(`kcol_dense.csv`). A missing entry is scaled by rows from the same type and row width (at most 4× the rows), or taken
from the type's one-token time times a family member's growth with k; a type with neither is not offered at that
shape. Today that leaves Q8_0 (timed only at the 320-wide hyper-connection rows), Q6_K and Q5_K at the 2,560-wide
shapes, and Q4_0/Q5_0 outside the shared expert's down projection: C323 times exactly these.

**Quality.** A type's error on Flash-Next's own tensors, measured ([bench/pascal_mix/qerr.py](../../../bench/pascal_mix/qerr.py),
[run/qerr.sh](run/qerr.sh)): every dense matrix of 16 sampled layers (0, 1, 3, 4, 8, 11, 12, 16, 20, …, 44, 47) and
the output layer, in seven files that hold different types for the same tensors (Swift Q2_0, IQ3_XXS, IQ3_S, IQ2_XS
as published, pmix-q2-fast, pmix-q2-spine6, pmix-iq3-fast), dequantized and compared with the Swift-1.5 Q8_0 (the
reference of every KLD here), weighted by FN4's importance matrix (the mean squared activation of each input column):
noise = Σ_j imp_j Σ_i (w_ij − ref_ij)², relative to the same sum of ref². 485 tensor-file pairs
([qerr/qerr.tsv](qerr/qerr.tsv)); a two-way fit, log rel = a(tensor) + c(type), gives each type's factor across
tensors ([qerr/types.tsv](qerr/types.tsv)):

| Type | Bits a weight | Error against IQ4_XS | Pairs | Fit residual (log2) |
|---|---:|---:|---:|---:|
| Q6_K | 6.56 | 0.058 | 130 | 0.38 |
| Q5_K | 5.50 | 0.23 | 18 | 0.24 |
| Q5_0 | 5.50 | 0.29 | 18 | 0.17 |
| Q4_K | 4.50 | 0.86 | 45 | 0.16 |
| IQ4_XS | 4.25 | 1 | 116 | 0.20 |
| IQ4_NL | 4.50 | 1.01 | 38 | 0.12 |
| Q4_0 | 4.50 | 1.15 | 4 | 0.05 |
| IQ3_S | 3.44 | 2.6 | 34 | 0.17 |
| Q3_K | 3.44 | 3.6 | 28 | 0.10 |
| Q2_0 | 2.25 | 20.7 | 22 | 0.06 |

About 4× per bit, as uniform rounding predicts; Q4_K's 4.5 bits buy only 14 % less error than IQ4_XS's 4.25, and Q5_0
(no super-block) is a little worse than Q5_K at the same bits. Layer 1 alone (`ple_key`'s layer, measured afterwards:
[qerr/types_layer1.tsv](qerr/types_layer1.tsv)) gives the same factors within its spread. A pick's KLD change is
predicted as a gain times its change in noise; with one gain for every class, calibrated on spine6 (`vmix.py calib`:
pmix-q2-fast → pmix-q2-spine6 changed the agent KLD by −0.0233 for −3,264 units of noise, so 7.13e-6 a unit;
[calib/](calib/)), until C326's arms give a gain per class.

**First picks** (`vmix.py select` on today's tables, [design/](design/): `plan.tsv` per tensor with its start and new
type, time and source; `types.txt`, `take.txt`, `retype.txt` for pascal-mix's `run/build.sh`; `summary.tsv`). The
times are the dense products' (attention, recurrent, shared expert, `ple_key`) in ms a pass, bench basis; the routed
experts, hyper-connections, router and output layer are unchanged:

| Pick | Objective | T(4), ms | T(1), ms | MB on the cards | Predicted KLD agent / wiki / code | What changes |
|---|---|---:|---:|---:|---|---|
| pmix-q2-fast (start) | | 36.44 | 24.09 | | 0.2792 / 0.4264 / 0.1080 (measured) | |
| `vmix-q2-time` | least E[T], start's bytes and KLD | **34.27** | 24.21 | −0.1 | −0.0009 | IQ4_XS → Q4_K ×109; Q5_K/Q6_K → IQ4_XS ×10 and shared down Q5_0 → Q4_0 ×14 pay for the bytes |
| `vmix-q2-kld` | least KLD, start's bytes and E[T] | 35.72 | 23.91 | −0.1 | **−0.0126** / −0.0029 / −0.0021 | the dense and shared tensors left in Q2_0 raised (Q2_0 → IQ4_XS ×7, → Q4_0 ×9, → Q5_0 ×2); IQ4_XS → Q4_K ×44; paid by Q4_K → IQ4_XS ×21, Q5_K/Q6_K → IQ4_XS ×12, shared down Q5_0 → Q4_0 ×13 |
| `vmix-q2-kld-b100` | least KLD, +100 MB | 36.40 | 25.77 | +99.9 | −0.0139 | the same, with more Q4_K and Q5_0 |

- The pass-time lever is small on today's types: −2.1 ms of a ~163 ms 4-token pass (Q591's T(4) with the 32 KB
  budget), because IQ4_XS and Q4_K are within ~5 % of each other at k = 4 and the types that might be faster there
  (Q8_0's rows, 2.2× faster with the right launch shape at the hyper-connection shape) have no times at the dense
  shapes yet. With spare bytes the time stops improving at 34.1 ms.
- **The quality lever is larger:** at the start's bytes and pass time the model predicts more than half of spine6's
  agent gain (−0.0126 against −0.0233 measured for est. +9.6 ms a token), because pmix-q2-fast keeps 2-bit dense and
  shared tensors (Q2_0's error is 20× IQ4_XS's) whose bytes are few. This rests on one pooled gain and on the noise
  model; building `vmix-q2-kld` and scoring it tests both (Q167 accepted predictions within two standard errors).
- P(1) = 0.011, so the k = 1 column hardly counts; on the bench the one-token times are the k-column kernels', which
  the model replaces by its one-token kernels (ggml / bench at one token is 0.66: [times_kshape_ratio.txt](times_kshape_ratio.txt)).

## 4. Q888 (C323): the types the selector lacks, on one P106

[run/rig_types.sh](run/rig_types.sh) runs `bench/mmvq_pascal`'s `mmvq_kcol` built with `FMT=1`
([build_kcol.sh](../../../bench/mmvq_pascal/build_kcol.sh)): Q8_0, Q6_K, Q5_K, Q5_0 and Q4_0 at every dense shape of
Flash-Next where the Swift Q2_0 census has none of them (`attn_qkv` 2,560 → 10,240, `attn_gate` 2,560 → 6,144, `attn_q`
2,560 → 12,288, `ssm_out` 6,144 → 2,560, the shared expert's 2,560 → 640 and 640 → 2,560), beside the census types
at the same shapes, at 1–8 tokens, every launch shape; each call is the activation quantizer and the product, and
every variant's rows are checked against a double-precision reference. ~20 minutes on one idle card. Its CSV goes
straight into `vmix.py times --bench`, and `select` reruns.

```bash
# on the build machine: bench/mmvq_pascal/build/mmvq_kcol with the new shapes
FMT=1 bench/mmvq_pascal/build_kcol.sh
# on the rig (nothing else running), then fetch the CSV
scp bench/mmvq_pascal/build/mmvq_kcol bench/mmvq_pascal/build/BUILDINFO_kcol.txt rig:/tmp/c323/
ssh rig "MMVQ=/tmp/c323/mmvq_kcol OUT=/tmp/c323/out setsid nohup bash -s > /tmp/c323/rig_types.out 2>&1" < run/rig_types.sh
scp rig:/tmp/c323/out/kcol_fmt.csv . && ssh rig 'rm -rf /tmp/c323'
```

**On one P106 (2026-11-04):** the run on card 0 of the second rig, nothing else running: `mmvq_kcol` exited 0 and all
3,008 rows (every type, shape, token count and launch shape) match the double-precision reference
([rig/kcol_fmt.csv](rig/kcol_fmt.csv), the bench's own table [rig/kcol_fmt.txt](rig/kcol_fmt.txt)). The selector's
`vmix.py times --bench rig/kcol_fmt.csv` and `select` are the next step.

## Commands

The builds (CPU only: pinned, nice 19, idle I/O priority; ~10–25 min a file, most of it writing ~11–38 GB):

```bash
# from a folder holding this repository as tools/; header dumps (bench/file_choice/dump.py) of pmix-q2-fast,
# pmix-q2-spine6, pmix-iq3-fast, swift-iq2_xs and swift15-q8_0 in DUMPS
export REPO=$PWD/tools DUMPS=<dumps> BASE=<pmix-q2-fast first shard> SPINE6=<pmix-q2-spine6 first shard>
export IQ3=<pmix-iq3-fast first shard> IQ2=<Swift IQ2_XS first shard> Q8=<Swift-1.5 Q8_0 first shard>
export MODELS=<output folder> WORK=<scratch> TABLE=<the n-gram table shard> CPUS=10,11,22,23
setsid -f nohup bash tools/docs/data/2026-11-04_fmt_mixes/run/chain.sh >> chain.log 2>&1 < /dev/null
python3 -I tools/bench/pascal_mix/galloc_test.py $WORK/switches.tsv <scratch>    # the solver's self-test
```

The scores, with pascal-mix's runner (FN2's `kld.sh` against FN1's bases, the bases' split, each file read where it
lies), the same five cards as pmix-q2-fast's own score; the list holds `<name> <first shard> <build marker>` lines:

```bash
export ENV=<FN1's environment file, GPUS_ALL the five cards> QUALITY=<2026-11-03_flash_next_quality's run/>
export BASES=<FN1's kld folder> OUT=<score folder> HOT=<scratch> LINKS=<folder for links> SCORE_TAG=fmt
export TABLE_FAST=<the table shard on the fast disk> DONE=<marker folder> LIST="$(cat <job list>)"
export SCORE=tools/docs/data/2026-11-04_pascal_mix/run/score_direct.sh
setsid -f nohup bash tools/docs/data/2026-11-04_pascal_mix/run/score_all.sh >> score.log 2>&1 < /dev/null
# one text at a time: TEXTS=agent (or wiki, code) with its own DONE folder; FN2's kld.sh skips texts already scored.
# Beside other jobs on the cards: QUALITY=<a copy of that run/ with pascal-mix's run/wait_shared.sh appended to
# common.sh> MIN_FREE_MIB=6000,7000,12000,12000,12000 (a floor per card; the split stays -fit on)
python3 -I tools/docs/data/2026-11-04_fmt_mixes/run/kld_table.py --meta kld_agent.tsv --logs <OUT> --text wiki > kld_wiki.tsv
# Code text (2026-11-10), on four cards: ENV=<a copy of the environment file with GPUS_ALL the four cards>, the
# shared-card QUALITY above with a floor per card, and the N2P folder's kld.sh as the driver (score_all.sh one file at a
# time, the cards' memory sampled beside it, DONE/finished at the end). With the files on a hard disk, readahead.sh
# beside it reads the next file into the page cache while the current one is scored.
export TEXTS=code GPUS=<the four cards> MIN_FREE_MIB=8000,10000,20000,20000 DONE=<marker folder for code>
export SCORE_ALL=tools/docs/data/2026-11-04_pascal_mix/run/score_all.sh
setsid -f nohup bash tools/docs/data/2026-11-04_n2p_flash/run/kld.sh >> score_code.log 2>&1 < /dev/null
LOG=score_code.log END=$DONE/finished SLOW=<path prefix of the hard disks> CARD=<one of the cards> TEXT=code \
  setsid -f nohup bash tools/docs/data/2026-11-04_fmt_mixes/run/readahead.sh >> readahead.log 2>&1 < /dev/null
# The start on the same four cards (Q1185): the two exports and the kld.sh line above with its own DONE folder and a
# one-line list, LIST="pmix-q2-fast-4card <pmix-q2-fast first shard>" (only one text of that name is scored, so the
# end marker's status is 1). Then a copy of the score folder, OUT4, in which that log stands under the start's name:
# the code table, the three-text table and the solve read the start's code score from there.
mkdir <OUT4>
for n in $(awk -F'\t' 'NR>1{print $1}' kld_agent.tsv); do for t in agent wiki code; do cp -L <OUT>/$n.$t.log <OUT4>/; done; done
cp -L <OUT>/pmix-q2-fast.code.log <OUT4>/pmix-q2-fast-5card.code.log
cp <OUT>/pmix-q2-fast-4card.code.log <OUT4>/pmix-q2-fast.code.log
python3 -I tools/docs/data/2026-11-04_fmt_mixes/run/kld_table.py --meta kld_agent.tsv --logs <OUT4> --text code > kld_code.tsv
sed 's/^pmix-q2-fast-4card\t/pmix-q2-fast\t/' <OUT>/mem_peak.tsv > mem_peak_code.tsv
bash tools/docs/data/2026-11-04_fmt_mixes/run/kld_texts.sh . > kld_texts.tsv    # kld_agent, kld_wiki, kld_code in .
python3 -I tools/bench/pascal_mix/galloc.py collect kld.tsv pmix-q2-fast=<OUT4>/pmix-q2-fast guiq35=<OUT4>/guiq35 ...
python3 -I tools/bench/pascal_mix/galloc.py solve --switches switches.tsv --kld kld.tsv --out solve.tsv \
  --budgets 37470389760,44.3e9,47e9    # the start's bytes on the cards, then the two of the allocation
python3 -I tools/bench/pascal_mix/galloc.py solve --switches switches.tsv --kld kld.tsv --out solve_ms.tsv --ms-budget 80
# does a whole build add up: pmix-iq3-fast's change against its switches' summed changes (header dumps from DUMPS)
python3 -I tools/docs/data/2026-11-04_fmt_mixes/run/additivity.py solve-switches.tsv \
  tools/docs/data/2026-11-04_pascal_mix/kld.tsv $DUMPS/pmix-q2-fast.json $DUMPS/pmix-iq3-fast.json > additivity.tsv
# the pick at the start's bytes as one file (the build exports above; MODELS on a disk with room for two copies, ~76 GB,
# while the steps run), then its fit at 64k from the headers and its header copies for a rig
REPO=$PWD/tools SWITCHES=tools/docs/data/2026-11-04_fmt_mixes/switches.tsv \
  PICKS=guiq30,denseiq31,guiq21,denseiq32,guiq23 NAME=gmix-q2 PREFIX=<the start's file prefix> \
  DONORS="pmix-iq3-fast=$IQ3 swift-iq2_xs=$IQ2" bash tools/docs/data/2026-11-04_fmt_mixes/run/build_pick.sh
REPO=$PWD/tools OUT=<folder> FILES="pmix-q2-fast=<its three shards, comma separated> gmix-q2=<its three shards>" \
  bash tools/docs/data/2026-11-04_fmt_mixes/run/pick_fit.sh
```

The selector (standard library; `qerr.py` needs numpy and gguf-py):

```bash
REF=<Swift-1.5 Q8_0 first shard> IMATRIX=<FN4's imatrix> FILES="swift-q2_0=<shard> pmix-q2-fast=<shard> ..." \
  OUT=qerr VENV=<a Python 3.12 venv with numpy> GGUF_PY=<llama.cpp tree>/gguf-py MOUNTS="<dirs>" bash run/qerr.sh
D=tools/docs/data
python3 -I tools/bench/pascal_mix/vmix.py times --bench $D/2026-11-04_kcurve_p106/kcol_dense.csv \
  --ggml-est $D/2026-11-04_kshape/ggml_est.tsv --perf $D/2026-11-04_kcurve_p106/perf_dense.tsv \
  --perf $D/2026-11-04_kcurve_p106/perf_cov.tsv [--bench kcol_fmt.csv] > times.tsv
python3 -I tools/bench/pascal_mix/vmix.py pk --metrics <metrics at the arm's start> <metrics at its end>
python3 -I tools/bench/pascal_mix/vmix.py calib --start pmix-q2-fast.json --qerr qerr --kld kld.tsv \
  --arm pmix-q2-spine6=pmix-q2-spine6.json [--arm s6gdn_qkv=<its dump> ...] > calib.tsv
python3 -I tools/bench/pascal_mix/vmix.py select --start pmix-q2-fast.json --published swift-q2_0.json \
  --times times.tsv --qerr qerr --calib calib.tsv --out vmix --name vmix-q2
# build it like a pascal-mix: PLAN=vmix NAME=<prefix> BASE=<Swift Q2_0> ... bash ../2026-11-04_pascal_mix/run/build.sh
```

## Files

| File | What it is |
|---|---|
| [switches.tsv](switches.tsv) | Q219's 30 switches and the start (`galloc.py switches`): tensors, type changes, bytes and est. ms added |
| [spine_switches.tsv](spine_switches.tsv) | C326's 8 arms, the same columns |
| [kld_agent.tsv](kld_agent.tsv) | agent-text KLD of the 30 single-switch files, the control `dniq23`, `pmix-q2-fast` and spine6: ±, change, same top token, class |
| [kld_wiki.tsv](kld_wiki.tsv) | wiki-text KLD of the same files: the same columns, then the change from the control `dniq23` and the class against it |
| [kld_code.tsv](kld_code.tsv) | code-text KLD of the same files, the wiki table's columns; every score, the start's included, from the same four cards |
| [kld_texts.tsv](kld_texts.tsv) | the three texts side by side: each file's change from the start and its class on agent, wiki and code text, and on how many texts it gains and loses |
| [solve_kld.tsv](solve_kld.tsv) | the solve's scores (`galloc.py collect`): each file's KLD, ± and same top token per text, the start's code score from the four cards |
| [solve.tsv](solve.tsv) | `galloc.py solve`'s picks at the start's bytes, 44.3 GB and 47 GB: switches, bytes, damage, est. ms, predicted KLD per text and the summed switches' ± |
| [solve-switches.tsv](solve-switches.tsv) | each switch's cost in the solve: bytes, est. ms, damage, its KLD change per text, damage per GB |
| [additivity.tsv](additivity.tsv) | `pmix-iq3-fast`'s measured change against the sum of its 16 switches' changes, per text |
| [pick_fit_64k_cards.txt](pick_fit_64k_cards.txt) | `pmix-q2-fast` and `gmix-q2` on the two nine-card 64k lines: layers and MiB per card |
| [fit-gmix-q2.tsv](fit-gmix-q2.tsv) | `fit.py`'s grid for `gmix-q2`: every context, cache type, micro-batch and draft, whole layers and with experts moved |
| [mem_peak_code.tsv](mem_peak_code.tsv) | the code-text runs' peak memory per file and card (the cards' index on the machine), MiB: the 31 files and the start |
| [run/kld_table.py](run/kld_table.py) | one text's table from llama-perplexity's logs (standard library) |
| [run/kld_texts.sh](run/kld_texts.sh) | `kld_texts.tsv` from the three tables |
| [run/readahead.sh](run/readahead.sh) | beside a one-text scoring run: the next file read from a slow disk into the page cache while the current one is scored |
| [times_kshape.tsv](times_kshape.tsv) | the P106 product times `vmix.py` uses (type, row width, rows, tokens, µs, source) |
| [pk_s33_mtp2d.txt](pk_s33_mtp2d.txt) | P(k) from S33's `mtp2d` arm (`vmix.py pk`) |
| [times_kshape_ratio.txt](times_kshape_ratio.txt) | the ggml / bench ratio per token count that puts the tables on one basis |
| [qerr/](qerr/) | `qerr.tsv` (each tensor's importance-weighted error per file and type), `types.tsv` (the type factors), `types_layer1.tsv` |
| [calib/](calib/) | the gain from spine6 per text (`vmix.py calib`) and the scores it used (`kld_pmix.tsv`, pascal-mix's) |
| [design/](design/) | the first picks: `plan.tsv`, `summary.tsv`, `types.txt`, `take.txt`, `retype.txt` each |
| [run/chain.sh](run/chain.sh), [run/build_switches.sh](run/build_switches.sh) | the single-switch builds, C326's arms then Q219's switches |
| [run/build_pick.sh](run/build_pick.sh) | a chosen set of switches as one file |
| [run/additivity.py](run/additivity.py) | do the single switches' changes add up: a whole build against its switches |
| [run/pick_fit.sh](run/pick_fit.sh) | a pick's fit at 64k from its headers beside the start's, `fit.py`'s grid and header copies for a rig |
| [run/spine_arms.tsv](run/spine_arms.tsv) | C326's arms (`galloc.py switches --spec`) |
| [run/qerr.sh](run/qerr.sh) | the error measurement in a Python container |
| [run/rig_types.sh](run/rig_types.sh) | C323 on one P106 |

## Provenance and licence

The files composed here are derived from Swift-Qwen3.8-Flash-Next GSQ-RCO Q2_0, IQ2_XS and IQ3_XXS as published, the
Pascal mixes made from them ([pascal-mix](../2026-11-04_pascal_mix/README.md)) and ukisai's
`Swift-1.5-Qwen3.8-Flash-Next-GGUF` Q8_0 (revision `0142554c`); none is redistributed here, and their licences are the
models' own. The tables are derived from the files' headers and tensors and from the k-curve, kshape and S33 runs of
this project. The P(k) counters are llama-server's `/metrics` of S33's `mtp2d` arm.
