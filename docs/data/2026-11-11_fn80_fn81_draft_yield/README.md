# FN80 and FN81: the MTP draft at 3 to 7 tokens, and token trees priced offline (2026-11-11)

Registry: [Q1396](../../../quants/experiments.md) (FN80: what a draft of 3 to 7 tokens yields) and Q1397 (FN81: token
trees for the MTP draft, priced on FN80's dumps). They answer for Q1394 (C797: longer drafts with the read-once 5–8-token
verify), Q614 (C87: token trees) and Q1398 (C799: a siblings tree). Tools: [bench/draft_count](../../../bench/draft_count/README.md)
and [bench/tree_replay](../../../bench/tree_replay/README.md), checked on a CPU in
[2026-11-11_c798_draft_count_tools](../2026-11-11_c798_draft_count_tools/README.md).

Flash-Next decodes with its MTP draft: the draft guesses a few tokens, and the main model checks them all in one verify
pass of draft + 1 tokens. The release preset drafts 3. This run counts how many tokens a step a draft of 3, 4, 5, 6 and
7 keeps on recorded agent requests. It then asks whether a tree, with a second guess beside the first at some
positions, would keep more than a longer plain draft of the same verify width. These are counts, not times: acceptance
depends on the models, the sampler and the text, not on the card. The cost of a step at each width is the rig's
measurement (R121); until then the costs below are estimates from earlier rig profiles.

## Result

- **A longer draft keeps more tokens a step up to 6, and 7 adds nothing over 6:**

  | Draft | Tokens a step, each load | Against 3 | 95 % over requests | 95 % over traces | Share accepted |
  |---|---|---:|---|---|---:|
  | 3 | 3.2401 and 3.2401 | | | | 0.7578 |
  | 4 | 3.7028 and 3.7028 | +0.4627 (+14.28 %) | +0.3661 to +0.5539 | +0.3663 to +0.5562 | 0.6877 |
  | 5 | 4.1257 and 4.1257 | +0.8855 (+27.33 %) | +0.6984 to +1.0408 | +0.6928 to +1.0452 | 0.6359 |
  | 6 | 4.5037 and 4.5037 | +1.2636 (+39.00 %) | +1.1011 to +1.4140 | +1.1003 to +1.4425 | 0.5975 |
  | 7 | 4.5024 and 4.5024 | +1.2623 (+38.96 %) | +1.1186 to +1.4153 | +1.1276 to +1.4472 | 0.5138 |

  The two loads of each length wrote the same 66 replies and the same counts, so their spread is 0.0000. Different
  lengths wrote different text (2 to 6 of the 66 replies the same between lengths), so the comparisons rest on the
  intervals over requests and traces.
- **Acceptance by position does not fall along the draft:** each position's conditional acceptance is between 0.8160 and 0.8807 at every length (table below). On the same steps, each added position is worth less than the one before only
  because fewer steps get that far.
- **On paper, a 3-token draft stays best at today's estimated costs.** A longer draft pays only if one more verify
  column, plus one more drafting pass of 3.645 ms, costs less than what its extra tokens are worth. Against the 3-token
  draft, the largest column cost at which each length still pays (est.):

  | Draft | Break-even column, ms | Its 95 % range | Est. tok/s at a column of 10 ms | at 15.36 ms (R41, 1 to 4 tokens) | at 21.36 ms (R41, the fifth token) |
  |---|---:|---|---:|---:|---:|
  | 3 | | | 25.13 | 25.13 | 25.13 |
  | 4 | 14.77 | 10.92 to 18.39 | 25.97 | 25.03 | 24.06 |
  | 5 | 13.97 | 10.25 to 17.06 | 26.41 | 24.72 | 23.06 |
  | 6 | 13.11 | 10.96 to 15.11 | 26.51 | 24.22 | 22.08 |
  | 7 | 8.91 | 7.48 to 10.43 | 24.54 | 21.97 | 19.67 |

  On v3, a column cost 15.36 ms between one and four tokens and 21.36 ms for the fifth (R41). At those costs no
  longer draft beats 3; 4 comes within 0.4 % (est.). **If the read-once 5–8-token kernels (C729) bring a column below
  about 13 ms on the rig, 5 or 6 tokens win**, by an est. 5.5 % for 6 at 10 ms. 7 tokens needs a column under 8.91 ms.
  On the same text, the 7th token against the 6th pays only below 6.50 ms. The est. tok/s is tokens a step over an
  estimated step time, a diagnostic for picking the rig test, not an agentic speed.
- **Every token tree loses to the plain draft of the same verify width (FN81).** A second guess at a position is worth
  something only on steps whose first rejection falls there, and those are few. At position 1 the main model's token
  is the draft's next candidate in 0.5278 of the 970 rejections there. That adds 0.0798 tokens over all 6,420 steps.
  Extending the plain draft by one token adds P_{L+1}, which is 0.2798 even at the 7th position. So for every shape,
  the tree keeps fewer tokens a step than the chain with as many verified tokens. Examples: siblings 2x1 on 6 (7
  tokens verified) 4.4025 against 4.6025; siblings 2x1 on 3 (4 verified) 3.1910 against 3.5891; full 2x2 on 3 (7
  verified, est.) 3.2854 against 4.6025. **Trees are not worth building for this draft (C87's tree half, C799).**
  - The tree saves drafting passes, but that does not close the gap. With a one-token step of an est. 71.90 ms and a
    column of 0.214 of it, siblings 2x1 on 6 would match the 7-token chain only if a drafting pass cost about 11.2 ms
    (est.), three times R86's 3.645 ms.
- **Memory on the newer cards grows by 74, 188 and 188 MiB from a draft of 3 to 7** (the RTX 3080 12 GB and the two
  RTX 3090s: 9,547 to 9,621, 16,322 to 16,510 and 16,284 to 16,472 MiB after loading). The MTP line sizes its
  recurrent snapshots to the draft length (`n_rs_seq` = `--spec-draft-n-max`), so no step restored a checkpoint at any
  length (0 in 41,638 verify steps). Whether 8 snapshots fit the rig's 6 GB cards and 4 GB host is for R121 and C797.
- **Every check passed:** 70 of 70 (the replay's counters = the trace's per step = `/metrics` per position = the dump's
  steps, no restore, every request matched to its task), and the dumps' check found 0 problems in 6,556 lines.

### Acceptance by position

P_i is the share of steps that drafted i or more tokens and kept the first i; c_i is the acceptance of position i
given every earlier one was kept (pooled over each length's loads, from the trace lines; the server's counter per
position agrees exactly).

| Draft | P1 | P2 | P3 | P4 | P5 | P6 | P7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.8807 | 0.7507 | 0.6417 | | | | |
| 4 | 0.8664 | 0.7306 | 0.6236 | 0.5293 | | | |
| 5 | 0.8701 | 0.7258 | 0.6075 | 0.5253 | 0.4491 | | |
| 6 | 0.8612 | 0.7371 | 0.6176 | 0.5230 | 0.4528 | 0.3903 | |
| 7 | 0.8481 | 0.6916 | 0.5710 | 0.4774 | 0.3949 | 0.3313 | 0.2771 |

| Draft | c1 | c2 | c3 | c4 | c5 | c6 | c7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.8807 | 0.8522 | 0.8545 | | | | |
| 4 | 0.8664 | 0.8434 | 0.8537 | 0.8491 | | | |
| 5 | 0.8701 | 0.8342 | 0.8377 | 0.8654 | 0.8553 | | |
| 6 | 0.8612 | 0.8557 | 0.8381 | 0.8473 | 0.8661 | 0.8636 | |
| 7 | 0.8481 | 0.8160 | 0.8254 | 0.8364 | 0.8272 | 0.8392 | 0.8368 |

The 7-token loads wrote text the draft found harder at every position. On their own steps cut to 3, they keep 3.1059
tokens a verify step against the 3-token loads' 3.2697. That is why 7 matches 6 overall although its 7th position is
kept 0.2771 of the time. [breakeven.md](breakeven.md) has the same-text view: each load's steps cut to every shorter
length. On the 7-token loads' steps, positions 4 to 7 add +0.4740, +0.3914, +0.3280 and +0.2736 tokens a verify step.

By context length ([tables/by_context.tsv](tables/by_context.tsv)), tokens a step does not fall from requests under
4,096 tokens to those of 8,192 and more. At 6 tokens: 4.6448, 4.4779 and 4.4064 for under 4,096, 4,096 to 8,191 and
8,192 or more (the last bin holds 6 requests a load).

### The token trees (FN81)

Both 7-token loads' dumps (`LLAMA_SPEC_DUMP_K=3`: the draft's 3 most probable candidates at every drafted position, plus
the drafted token) went through `tree_replay.py --mode factorized`. Of 6,556 steps, 6,420 had the continuation known to
the 7th position. The two dumps hold the same steps, so each step counts twice. The means are one load's, and the
standard errors below are √2 too small; every tree-minus-chain difference is still more than 19 corrected standard
errors from 0.

- **A siblings tree** is the walk cut to L tokens with w − 1 leaves beside each of its first m tokens. The leaves are the
  draft's next candidates, from the passes that drafted the walk, so the tree needs no extra drafting pass. Its value
  is exact under probabilistic verification: on the walk's first rejection, the main model's draw from the residual is
  matched against the leaves.
- **A full tree** verifies every path over the top w at each of the first m positions. It is an estimate: below an
  off-walk candidate, the dump has only the walk's candidates.

| Shape | Exact | Verified tokens | Tree | Chain of as many | Tree − chain | se |
|---|---|---:|---:|---:|---:|---:|
| siblings 2x1 on 3 | yes | 4 | 3.1910 | 3.5891 | −0.3981 | 0.0079 |
| siblings 2x2 on 3 | yes | 5 | 3.2667 | 3.9879 | −0.7212 | 0.0143 |
| siblings 3x1 on 3 | yes | 5 | 3.2162 | 3.9879 | −0.7717 | 0.0136 |
| full 2x2 on 3 | est. | 7 | 3.2854 | 4.6025 | −1.3171 | 0.0243 |
| siblings 2x1 on 4 | yes | 5 | 3.6688 | 3.9879 | −0.3190 | 0.0077 |
| siblings 2x1 on 5 | yes | 6 | 4.0676 | 4.3227 | −0.2551 | 0.0074 |
| siblings 2x1 on 6 | yes | 7 | 4.4025 | 4.6025 | −0.2000 | 0.0071 |

Tokens a verify step (accepted + 1). Every shape is in [tree/summary.md](tree/summary.md) and
[tree/shapes.csv](tree/shapes.csv); shapes with more than 7 verified tokens have no chain to compare with in a 7-token
dump.

What a leaf is worth, by position ([tree/siblings.csv](tree/siblings.csv)): of the steps whose walk is first rejected
at position j, the share where the main model's token is the draft's next candidate (top 1 other), or among the next 2
or 3.

| Position | Rejections there | Top 1 other | Top 2 | Top 3 |
|---:|---:|---:|---:|---:|
| 1 | 970 | 0.5278 | 0.6948 | 0.7134 |
| 2 | 1,010 | 0.4812 | 0.6673 | 0.6752 |
| 3 | 776 | 0.4356 | 0.6108 | 0.6211 |
| 4 | 596 | 0.5034 | 0.6477 | 0.6544 |
| 5 | 508 | 0.4409 | 0.6220 | 0.6339 |
| 6 | 410 | 0.5122 | 0.6585 | 0.6634 |
| 7 | 354 | 0.4463 | 0.6045 | 0.6158 |

## The cost estimates (est., until R121)

R121 maps the v4 step at every width with the kernel merge's switches on. Until then, the costs come from two rig
profiles:

- **The 3-token step: 128.92 ms**, R86's served step on the v4 line (the night merge), the greedy request's median
  ([2026-11-11_r86_step_trace](../2026-11-11_r86_step_trace/README.md), `split_prof.md`).
- **A drafting pass: 3.645 ms**, R86's passes 1 and 2 (lead, GPU and turn: 3.67 and 3.62 ms, medians).
- **A verify column**, from R41's v3 pass times without the draft
  ([2026-11-10_release_profile](../2026-11-10_release_profile/README.md)). One token takes 50.925 ms (51.10 and 50.75
  in two loads), four take 97.01 (97.01 and 97.01) and five take 118.365 (118.38 and 118.35). So a column costs
  15.36 ms on average from one to four tokens, and the fifth costs 21.36 ms.
- A draft of d tokens is priced as ms(d) = 128.92 + (d − 3) × (column + 3.645). [run/cost_readonce.tsv](run/cost_readonce.tsv)
  puts every column at 15.36 ms (the read-once kernels at every width); [run/cost_v3col.tsv](run/cost_v3col.tsv) puts
  the columns beyond 4 at 21.36 ms (v3). `draft_table.py --cost` adds est. tok/s
  ([tables/arms_cost_readonce.tsv](tables/arms_cost_readonce.tsv), [tables/arms_cost_v3col.tsv](tables/arms_cost_v3col.tsv)),
  and [run/breakeven.py](run/breakeven.py) gives the break-even columns.
- **For the trees, in one-token steps:** a one-token step is 128.92 − 3 × 15.36 − 3 × 3.645 = 71.90 ms. One more
  verified token is r = 15.36 / 71.90 = 0.214, or 21.36 / 71.90 = 0.297. A drafting pass is 3.645 / 71.90 = 0.0507.
- **Not counted:** the recurrent snapshots grow with the draft (8 at a draft of 7 against 4 at 3). R41 put 4.91 ms of the draft
  line's five-token pass (110.34 against 104.03 ms of GPU time) on the rollback snapshots.

## What ran

- **Build:** `pascal-p106-v4` at `08634ce00` (the v4 late merge), its rig tarball's `llama-server` (CUDA 12.9, sm_61)
  on newer cards, the driver compiling the PTX on the first load.
- **Model:** Qwen3.8-Flash-Next, UkisAI's Swift-1.5 GSQ-RCO Q2_0 with its MTP draft, on the line of
  [release/flash-next-v3/settings.env](../../../release/flash-next-v3/settings.env). The preset's sampler (temperature
  1.0, top-p 0.95, top-k 20) and backend sampling (`-bs`) were kept. Every load added probabilistic verification
  (`--spec-draft-sampling probabilistic --spec-draft-p-min 0`) and C675's end-of-turn stop
  (`LLAMA_SPEC_REJECT_STOP=1`). The draft length was `--spec-draft-n-max N` in place of the preset's 3. The loads also
  had `--metrics`, `LLAMA_TRACE=1` and the dump. The preset's nine-card split and keep-alive were left out; the split
  was `-ts 23861,23861,11681`, the two 3090s first, as CUDA lists them.
- **Cards:** an RTX 3080 12 GB and two RTX 3090s. One container per load; one load at a time.
- **Requests:** the v3 release check's `t1swift` arm, 66 requests in 12 traces (12 bug-fix tasks), contexts of 1,952 to 15,137 tokens. They were re-sent one at a time with their recorded seeds and reply
  limits by [bench/agent_check/replay.py](../../../bench/agent_check/replay.py).
- **Loads:** `d3 d4 d5 d6 d7 d7 d6 d5 d4 d3`, 19:54–21:31, about 10 minutes each. Every reply was whole (66 of 66 in
  each load).
- **Device sampling:** every token was drawn on the device (the server's count of device draws is 3 to 9 above the
  generated tokens in each load). There was 1 grammar rejection in each 3-token load and 0
  elsewhere.
- [run_lines.tsv](run_lines.tsv) has every load's exact environment and arguments.

## Files

| File | What |
|---|---|
| [tables/summary.md](tables/summary.md) | `draft_table.py`'s tables in one page: loads, arms, P_i and c_i, against 3, by context, checks |
| [tables/loads.tsv](tables/loads.tsv), [tables/arms.tsv](tables/arms.tsv), [tables/compare.tsv](tables/compare.tsv) | per load; per length with each load by value; each length against 3 with intervals |
| [tables/positions.tsv](tables/positions.tsv), [tables/by_context.tsv](tables/by_context.tsv), [tables/cut.tsv](tables/cut.tsv) | P_i and c_i per load; by context length; each load's steps cut to shorter drafts |
| [tables/requests.tsv](tables/requests.tsv), [tables/same_reply.tsv](tables/same_reply.tsv), [tables/check.tsv](tables/check.tsv) | per request (counts, reply MD5, per position); replies the same between loads; the 70 checks |
| [tables/arms_cost_readonce.tsv](tables/arms_cost_readonce.tsv), [tables/arms_cost_v3col.tsv](tables/arms_cost_v3col.tsv) | `arms.tsv` with est. tok/s on the two cost tables |
| [breakeven.md](breakeven.md) | the break-even column per length, est. tok/s at five column costs, the same-text view |
| [fit.tsv](fit.tsv) | each card's memory after loading, per load |
| [run_lines.tsv](run_lines.tsv) | each load's environment and arguments, as run |
| [tree/summary.md](tree/summary.md), [tree/shapes.csv](tree/shapes.csv), [tree/siblings.csv](tree/siblings.csv), [tree/chain.csv](tree/chain.csv), [tree/check.txt](tree/check.txt) | FN81: `tree_replay.py`'s tables on the two 7-token dumps |
| [run/cost_readonce.tsv](run/cost_readonce.tsv), [run/cost_v3col.tsv](run/cost_v3col.tsv), [run/breakeven.py](run/breakeven.py) | the est. step times and the break-even script |

## Commands

On the machine with the cards, a work folder holding this repository's `bench/draft_count`, `bench/tree_replay`,
`bench/agent_check/{replay,common}.py` and `release/flash-next-v3/settings.env` in `repo/`, the recorded `proxy.jsonl`
with its `.bodies.gz` in `text/`, and an env file setting the variables `run_draft_count.sh` names: `WORK`, `REPO`,
`BIN` (`llama-server`), `MODEL` (Swift Q2_0's first shard), `MODEL_DRAFT`, `LOG`, `GPUS` (the three cards' docker
device ids), `MOUNTS`, `JIT`, `NAME`, `PORT`, and `TS=23861,23861,11681`, `LOADS="d3 d4 d5 d6 d7 d7 d6 d5 d4 d3"`.

```bash
. ./fn80.env && setsid -f nohup nice -n 19 bash repo/bench/draft_count/run_draft_count.sh > console.txt 2>&1 < /dev/null
# when $WORK/done exists:
python3 -I repo/bench/draft_count/draft_table.py "$WORK" --out tables --ref d3
python3 -I repo/bench/draft_count/draft_table.py "$WORK" --out tables-readonce --ref d3 --cost cost_readonce.tsv
python3 -I repo/bench/draft_count/draft_table.py "$WORK" --out tables-v3col --ref d3 --cost cost_v3col.tsv
python3 -I breakeven.py tables > breakeven.md
# FN81, CPU only:
python3 -I repo/bench/tree_replay/tree_replay.py --check "$WORK"/dump-d7-1.jsonl "$WORK"/dump-d7-2.jsonl
python3 -I repo/bench/tree_replay/tree_replay.py "$WORK"/dump-d7-1.jsonl "$WORK"/dump-d7-2.jsonl --out tree-d7 \
  --mode factorized $(python3 -I repo/bench/tree_replay/gguf_markers.py "$MODEL") --r 0.214,0.297 --pass-cost 0.0507
```

`cost_readonce.tsv`, `cost_v3col.tsv` and `breakeven.py` are this folder's `run/`. `gguf_markers.py` printed the
tool-call and reasoning markers `--tool-open 248058 --tool-close 248059 --think-open 248068 --think-close 248069`.

## Provenance and licence

- The requests are the v3 release check's own recording of this project's agent check. The tables hold counters,
  task ids and reply hashes only.
- The model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next with its MTP draft, as in the release preset.
- Raw output (server logs, the replay's rows with their times, the dumps) is not published.
- The tables are this repository's tools' output, under the repository's licence.
