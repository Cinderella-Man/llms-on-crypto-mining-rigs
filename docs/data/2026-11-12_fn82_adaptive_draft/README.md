# FN82: the MTP draft's length chosen per step, priced offline (2026-11-12)

Registry: [Q1402](../../../quants/experiments.md) (FN82), for Q687 (`LLAMA_SPEC_ADAPT`, the length chosen per step)
and Q1394 (C797: longer drafts with the read-once 5–8-token verify). Data: FN80's dumps
([2026-11-11_fn80_fn81_draft_yield](../2026-11-11_fn80_fn81_draft_yield/README.md)). Tool:
[bench/draft_count/adapt_replay.py](../../../bench/draft_count/adapt_replay.py).

Flash-Next decodes with its MTP draft: the draft guesses a few tokens and the main model checks them in one verify pass.
FN80 counted what a fixed draft of 3 to 7 tokens keeps. A longer draft keeps more tokens a step, but it spends a verify
column and a drafting pass on every position, including those it is unsure of. `LLAMA_SPEC_ADAPT=1` (in
`pascal-p106-v4`) cuts each draft to the prefix that pays: from the draft's own probability of each token and a table
of what a verify pass of k tokens costs. This run replays that rule on FN80's recorded drafts and prices it against
every fixed length, under six cost tables. The replay counts tokens exactly; the times are estimates (est.) until R121
measures the v4 step at every width.

## Result

- **The switch as built, with a draft of up to 6 tokens, beats every fixed length under every cost table, on both
  recorded texts.** As built means `LLAMA_SPEC_ADAPT=1`, `--spec-draft-n-max 6`, its default S (`run`), widths
  (1, 3, 5, 9, 17) and scale (1). Est. tok/s against the best fixed length of each table, with the 95 % interval over
  requests (paired on the same steps):

  | Cost table | 6-token text: best fixed | as built | difference | 7-token text: best fixed | as built | difference |
  |---|---|---:|---|---|---:|---|
  | column 8 ms | 6: 27.943 | 28.359 | +0.417 (+0.257 to +0.603) | 6: 26.302 | 26.833 | +0.532 (+0.366 to +0.677) |
  | column 10 ms | 6: 26.961 | 27.728 | +0.766 (+0.616 to +0.958) | 5: 25.476 | 26.343 | +0.867 (+0.707 to +1.076) |
  | column 12 ms | 5: 26.153 | 27.196 | +1.042 (+0.859 to +1.270) | 5: 24.846 | 25.919 | +1.073 (+0.888 to +1.293) |
  | column 14 ms | 5: 25.520 | 26.752 | +1.232 (+1.060 to +1.446) | 4: 24.470 | 25.621 | +1.152 (+0.962 to +1.424) |
  | column 16 ms | 4: 25.162 | 26.387 | +1.225 (+0.960 to +1.507) | 4: 24.144 | 25.316 | +1.171 (+0.968 to +1.428) |
  | **v4 est.** | 3: 24.938 | 25.951 | **+1.014 (+0.603 to +1.424)** | 3: 24.114 | 24.815 | **+0.701 (+0.444 to +1.047)** |

  The 6-token text is FN80's `d6` load (3,537 verify steps), the 7-token text its `d7` load (3,278), each of the 66
  requests. Each length's second load holds the same steps and gives the same numbers ([replay/check.tsv](replay/check.tsv)).
- **Under the v4 est. table no fixed length beyond 3 pays, but the rule still gains, by +4.06 % and +2.91 % (est.).**
  The fixed lengths on the 6-token text: 23.046, 24.938, 24.305, 24.518 and 24.362 est. tok/s at 2 to 6 tokens. The
  rule keeps all 6 drafted tokens on the 1,840 steps where the draft is sure, 4 on 351, 2 on 1,020 and none on 326: 4.1660
  tokens a verify step at 5.0950 verify columns a step, against 4.5711 tokens at 6.9771 columns for a fixed 6.
- **Deciding while drafting would gain more (est.: not built).** The switch drafts all 6 tokens and then cuts, so every
  step pays 6 drafting passes (3.645 ms each). Its test for a length uses only the probabilities up to that length, so
  the drafter could stop at the first length that fails. The same rule taken while drafting drafts 5.0212 passes a
  step instead of 5.9771: 26.528 est. tok/s on the 6-token text (+1.591 over fixed 3, +1.215 to +1.972) and 25.413 on
  the 7-token text (+1.298, +1.044 to +1.635). With every width allowed it drafts 4.4080 passes a step: 27.127
  (+2.189, +1.817 to +2.590) and 26.086 (+1.972, +1.725 to +2.285). Registry Q1424.
- **The drafter's own stop, `--spec-draft-p-min 0.7` with a draft of up to 6, also beats every fixed length under the
  v4 est. table**, with no new code: 26.369 (+1.431, +0.876 to +1.918) and 25.086 (+0.972, +0.656 to +1.366). It
  stops drafting at the first position whose most probable candidate is under 0.7, so it saves drafting passes too.
  0.7 is the best of the six thresholds tried at most tables, so its numbers are picked on this data; the switch's
  numbers are its defaults.
- **At the preset's draft of 3 the rule gains nothing:** on the 3-token text it gives 25.300 est. tok/s against
  fixed 3's 25.378 under the v4 est. table (−0.077, −0.194 to +0.014).
- **What a perfect rule would give** (the draft cut to exactly what is accepted): 29.778 under the v4 est. table on the
  6-token text with every pass drafted, 31.107 with drafting stopped at the first rejection. The switch reaches about
  a fifth of that gap.
- **The draft's probability is a good guide where it is high and pessimistic where it is low**
  ([replay/calibration.tsv](replay/calibration.tsv)). At the first position of the 6-token text, the 1,895 drafted
  tokens with a probability of 0.99 or more were accepted 0.9905 of the time; the 547 under 0.5 (mean 0.2367) still
  0.5283 of the time. Probabilistic verification accepts a drafted token always when the main model rates it at least
  as high as the draft did, and otherwise with the ratio of the two, so a token the draft doubted often passes.

### The v4 cost table (est.)

R121 maps the v4 step at every verify width. Until it runs, [run/v4_cost.py](run/v4_cost.py) builds a table from
today's measurements ([v4_est.tsv](v4_est.tsv), its parts in [v4_est_parts.tsv](v4_est_parts.tsv)):

| Verify width k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V(k): the step without drafting, ms (est.) | 87.597 | 94.940 | 105.432 | 117.985 | 139.227 | 152.703 | 166.181 | 180.476 |
| the k-th column, ms (est.) | | 7.344 | 10.492 | 12.553 | 21.242 | 13.477 | 13.477 | 14.296 |
| the step at a draft of k − 1, ms (est.) | 87.597 | 98.585 | 112.722 | 128.920 | 153.807 | 170.928 | 188.051 | 205.991 |

- **The anchor is R86's v4 step:** 128.92 ms at a 3-token draft, less its three drafting passes of 3.645 ms, is V(4).
- **The products the v4 kernels change** move from R120b's served verify pass at 4 tokens by their single-product times
  at 1 to 8 tokens on a P106. Those times come from R120b's perf step for IQ4_XS, Q4_K, Q5_K and Q6_K, and from C805's
  rig run for Q3_K `ssm_out`. The arms are the ones the v4 candidate line runs: the read-once kernels up to 4 tokens
  (Q6_K with `_Q6K_MT`), and from 5 tokens the faster of `_RO8` and the plain kernel, as C809's rule will pick. C726's
  Q3_K kernel reads the 2,560-wide Q3_K products once at 2 to 8 tokens, but no P106 run has timed it past 4 tokens.
  Those products are scaled like the IQ4_XS and Q4_K read-once products of the same width (est.).
- **Every other part grows as in R41's v3 profile at 1, 4 and 5 tokens:** routed experts, shared expert,
  hyper-connections, router, Gated DeltaNet, attention, other operations, and hops and host. Past 5 tokens they
  continue at their 4-to-5 slope. The BF16 hyper-connections and the router jump at 5 tokens (a different kernel)
  and are held at their 5-token time.
- **The fifth column is the expensive one (est. 21.242 ms).** Of it: the hyper-connections +5.470 ms, the routed
  experts +4.530, the router +2.220, the Q3_K products +2.020, hops and host +1.130 (v3's jumps, R41). That is why
  no fixed length beyond 3 pays under this table. Whether v4 keeps the jump is for R121 to say (Q1425).
- **Not counted:** the recurrent snapshots grow with the draft. The MTP line sizes them to `--spec-draft-n-max`
  (FN80), so a rule with a draft of up to 6 may pay for 6 snapshots on every step. R41 put 4.91 ms of a five-token
  pass on them.

## For C797's rig session

- **One more arm:** `LLAMA_SPEC_ADAPT=1 --spec-draft-n-max 6` with
  `LLAMA_SPEC_ADAPT_COST=87.597,94.940,105.432,117.985,139.227,152.703,166.181` (V(1) to V(7) above) and
  `LLAMA_SPEC_ADAPT_DRAFT_MS=21.87` (6 drafting passes), the switch's other defaults. Add probabilistic verification and
  C675's stop, as on the other arms.
  - With every width (`LLAMA_SPEC_ADAPT_WIDTHS=1,2,3,4,5,6,7`) it prices +0.212 and +0.236 est. tok/s higher. That
    means seven verify shapes instead of four.
- **The no-code arm beside it:** `--spec-draft-p-min 0.7 --spec-draft-n-max 6`.
- **When R121's map lands,** rerun the replay with its table (`--table r121=FILE`, columns `k` and `ms`). It takes
  1.5 min on one CPU core. Then put the new V(1) to V(7) into the arm.

## How the replay works

- **One verify step a dump line.** The dump records the drafted tokens, how many the main model accepted (`n_acc`), and
  the draft's candidates with their probabilities at each position. `LLAMA_SPEC_ADAPT` reads the draft's probability of
  each drafted token from the same candidate list the dump records (`spec_push_p` and `spec_info_add_marginal` in
  `common/speculative.cpp`), so `p[i][pick[i]]` is the switch's input exactly, to the dump's 5 decimals.
- **Cutting a draft does not change which of its first tokens are accepted.** Each position's test depends only on the
  tokens before it. So a step cut to n drafted tokens keeps 1 + min(`n_acc`, n) tokens. Every arm is scored on the same
  steps of the same text. The fixed cuts reproduce FN80's `cut.tsv` to 4 decimals on all 40 lengths and loads.
- **The rule is a line-by-line port** of `common_spec_adapt_choose`, `_cost`, `_lengths` and `common_spec_adapt_state`
  at `pascal-p106-v4`'s tip, with the self-test's cases worked by hand. Its running speed-up (S `run`) carries over
  from request to request in the load's order, as the server's slot keeps it.
- **A step's est. time** is its drafting passes × 3.645 ms + V(verified + 1). The uniform tables put every column at
  8, 10, 12, 14 or 16 ms (FN80's model: V(k) = 128.92 − 3 × 3.645 + (k − 4) × column). The rule's own cost table is
  the same V(1) to V(n_max + 1), and its `draft_ms` is n_max × 3.645.
- **The arms:**
  - fixed 2 to 7;
  - the rule with a draft of up to 3 to 7, S `run` or `step`, default or every width, scale 0.5 to 2, as built or
    decided while drafting;
  - `--spec-draft-p-min` at 0.3 to 0.95;
  - the perfect cut.

  Every arm is in [replay/arms.tsv.gz](replay/arms.tsv.gz). The comparisons with intervals (2,000 resamples of the 66
  requests, the same resamples for every arm) are in [replay/compare.tsv](replay/compare.tsv).
- **What a replay cannot see:** cutting a step changes where the next one starts, so a live run writes other text. The
  replay holds the text fixed, as FN80's same-text view does. The rig run measures the rest.

## Files

| File | What |
|---|---|
| [main.md](main.md) | the main tables: est. tok/s per arm and cost table with intervals, and what each arm verifies and drafts, on the 6-, 7- and 3-token texts |
| [v4_est.tsv](v4_est.tsv), [v4_est_parts.tsv](v4_est_parts.tsv) | the v4 cost table (est.) and its parts' change from 4 tokens |
| [replay/arms.tsv.gz](replay/arms.tsv.gz) | every arm, cost table and load: tokens a verify step, columns a step, drafting passes a step, est. ms a step and tok/s, the lengths kept |
| [replay/compare.tsv](replay/compare.tsv) | the main arms against every fixed length, est. tok/s difference with its 95 % interval over requests |
| [replay/calibration.tsv](replay/calibration.tsv) | the draft's probability against acceptance, by position and probability bin |
| [replay/check.tsv](replay/check.tsv) | the fixed cuts against FN80's `cut.tsv`, and the two loads of each length against each other (45 of 45 OK) |
| [replay/summary.md](replay/summary.md) | `adapt_replay.py`'s own summary for every load and table |
| [run/fn82_run.sh](run/fn82_run.sh), [run/v4_cost.py](run/v4_cost.py), [run/main_table.py](run/main_table.py) | the run, the v4 cost table, the main tables |

## Commands

On any machine with Python 3 (no GPU; it ran on a build machine's CPU at the lowest priority in 1 min 34 s): a work
folder `WORK` holding this repository as `repo/` and FN80's dumps as `dumps/work/dump-d{3..7}-{1,2}.jsonl` (FN80's
loads with `LLAMA_SPEC_DUMP_K=3`; [its commands](../2026-11-11_fn80_fn81_draft_yield/README.md#commands)).

```bash
cp repo/docs/data/2026-11-12_fn82_adaptive_draft/run/fn82_run.sh "$WORK"/
WORK=$WORK bash "$WORK"/fn82_run.sh   # writes $WORK/tables, $WORK/replay, $WORK/main.md, then $WORK/done
```

The script runs, in order:

```bash
python3 -I repo/bench/draft_count/adapt_replay.py --selftest
python3 -I $F/v4_cost.py --r120b $D/2026-11-12_r120b_kern_speed --c805 $D/2026-11-12_c805_kcol_exact \
  --r41 $D/2026-11-10_release_profile --out tables
python3 -I repo/bench/draft_count/adapt_replay.py dumps/work/dump-d{3,4,5,6,7}-{1,2}.jsonl --out replay \
  --n-max 3,4,5,6,7 --fixed 2,3,4,5,6,7 --cols 8,10,12,14,16 --table v4est=tables/v4_est.tsv \
  --also 'adapt-n{N}-run-wdef-x1-online,pmin0.7-n{N}' --check-cut $D/2026-11-11_fn80_fn81_draft_yield/tables/cut.tsv
python3 -I $F/main_table.py replay > main.md
```

with `D=repo/docs/data` and `F=$D/2026-11-12_fn82_adaptive_draft/run`.

## Provenance and licence

- The dumps are FN80's: Qwen3.8-Flash-Next (UkisAI's Swift-1.5 GSQ-RCO Q2_0 with its MTP draft) on `pascal-p106-v4`
  `08634ce00`, answering the v3 release check's 66 recorded agent requests. The dumps hold token ids and probabilities
  and are not published; the tables hold counts and estimates only.
- The cost inputs are this repository's published rig measurements (R86, R120b, C805's R126, R41).
- The tables and scripts are this repository's, under its licence.
