# The rewritten IQ grid kernels on a P106 (C1146, Q1871, 2026-11-26)

What this measured: how fast the four rewritten multi-token routed gate/up kernels of Qwen3.8-Flash-Next's
better-quality file `gmix-q2` run on nine P106-090 cards. [R125](../2026-11-25_r125_gmix_step/README.md) found
`gmix-q2` about 18 ms a step slower than `pmix-q2-fast`, the file it was built from (Swift Qwen3.8-Flash-Next Pascal Q2
GGUF), almost all of it in the four-token routed gate/up products of its IQ2_XXS, IQ2_S, IQ2_XS and IQ1_M layers. Code
row C1138 rewrote those four kernels twice: first in CUDA C, then whole in hand-written PTX. Both versions give
results bit for bit the same as the old kernels, behind `GGML_CUDA_MTOK_IQ_FAST`: unset is the old kernels, `1` the
CUDA C ones, `2` the PTX ones, and the switch moves all four types at once ([IQ2_XXS](../2026-11-25_c1138_iq2xxs_mtok/README.md),
[IQ2_S](../2026-11-26_c1138_iq2s_mtok/README.md), [IQ2_XS](../2026-11-26_c1138_iq2xs_mtok/README.md),
[IQ1_M](../2026-11-26_c1138_iq1m_mtok/README.md), [the PTX notes](../../research/notes/ptx_kernels.md#the-four-kernels-summary)).
Here R125's group profile ran once at each setting, and R125's step pair ran at each setting, with `pmix-q2-fast` on
the same build for comparison. Registry row [Q1871](../../../quants/experiments.md).

## Result

**The PTX kernels take the four types' routed gate/up to half their old time, and IQ1_M's to Q2_0's time.** At four
tokens a layer now takes 0.464 ms in IQ2_XXS (old kernel 0.938), 0.434 in IQ2_S (0.984), 0.496 in IQ2_XS (1.038) and
0.268 in IQ1_M (0.549), against 0.262 in Q2_0. That removes 11.80 ms of the 15.91 ms that R125 estimated these four
types cost over Q2_0 (74 %). **The step is a gain: 99.68 ms against 111.34 at depth 0 (−11.66 ms, −10.47 %) and
114.13 against 125.91 at 8,192 (−11.78 ms, −9.36 %)**, two loads of each setting, with spreads of 0.05 to 0.25 ms.
`gmix-q2`'s step gap to `pmix-q2-fast` on the same build falls from +16.34 and +16.90 ms to +4.68 and +5.12 ms. The
CUDA C kernels get 81 % of the PTX kernels' gain. Every load wrote the same replies.

What limits the new kernels is the number of instructions they issue, not their shared-memory lookups or the bytes
they read. Their time per instruction is 0.96 to 1.13 times Q2_0's. IQ1_M, with the most lookup conflicts, runs at
Q2_0's rate. The IQ2 types still need 63 to 73 instructions per 32 weights against Q2_0's 43, which leaves est. 4.38 ms
a pass over Q2_0's rate.

### The routed gate/up product by type (group profile, `g0`, `g1`, `g2`)

ms a layer (one launch group of the fused gate/up product, 2560 × 640 per expert) on the repeated request at depth 0
(`d0p2`), all nine cards; R125's load (`gg`, the `eee615cdb` build) beside it ([types.md](types.md), which also has
depth 8,192 and both first requests):

| Type | layers | R125 | unset (old) | `1` (CUDA C) | `2` (PTX) | `2` − unset | `2` / Q2_0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| IQ2_XXS | 11 | 0.937 | 0.938 | 0.557 | 0.464 | −0.474 (−50.5 %) | 1.77 |
| IQ2_S | 7 | 0.978 | 0.984 | 0.535 | 0.434 | −0.550 (−55.9 %) | 1.66 |
| IQ2_XS | 4 | 1.042 | 1.038 | 0.598 | 0.496 | −0.542 (−52.2 %) | 1.90 |
| IQ1_M | 2 | 0.547 | 0.549 | 0.331 | 0.268 | −0.281 (−51.1 %) | 1.03 |
| Q2_0 | 24 | 0.271 | 0.262 | 0.262 | 0.262 | −0.000 | 1.00 |

At depth 8,192 (`d8192p2`) the same kernels take 0.463, 0.428, 0.500 and 0.271 ms a layer with `2`, against 0.936,
0.967, 1.050 and 0.552 unset and Q2_0's 0.267.

ms a verify pass (four tokens), the routed gate/up of the five types and the whole pass:

| | unset | `1` | `2` | saved by `1` | saved by `2` |
|---|---:|---:|---:|---:|---:|
| routed gate/up, depth 0 | 28.739 | 19.203 | 16.936 | 9.536 | 11.803 |
| the pass (sum of the parts), depth 0 | 95.938 | 86.349 | 84.059 | 9.589 | 11.879 |
| routed gate/up, depth 8,192 | 28.783 | 19.304 | 17.041 | 9.479 | 11.742 |
| the pass (sum of the parts), depth 8,192 | 109.196 | 99.750 | 97.449 | 9.446 | 11.747 |

Only the routed gate/up product changes: every other product of the pass is within 0.09 ms of the unset load
([products-g0-g2.md](products-g0-g2.md); `1` against unset and `2` against `1` in the other two `products-*` files).
The unset load repeats R125's to within 0.006 ms a layer for the four grid types. Q2_0's layers run 0.009 ms faster on
this build.

### The step (no profiler)

ms a step on the repeated request (pass 2), every load ([steps.md](steps.md); a step is one pass of the main model,
its generated tokens minus the accepted draft tokens):

| Context | unset (`u0`, `u0b`) | `1` (`u1`) | `2` (`u2`, `u2b`) | `pmix-q2-fast` (`p0`) |
|---:|---|---:|---|---:|
| 0 | 111.31, 111.37 | 101.91 | 99.61, 99.75 | 95.00 |
| 8,192 | 125.87, 125.95 | 116.12 | 114.01, 114.25 | 109.01 |

- **`2` against unset: −11.66 ms (−10.47 %) at depth 0 and −11.78 ms (−9.36 %) at 8,192, a gain** at both depths:
  the difference is far beyond the spreads (unset 0.05 and 0.08 ms, `2` 0.14 and 0.25).
- `1` against unset: −9.43 and −9.79 ms (one load, no class). `2` is 2.23 and 1.99 ms faster than `1` a step.
- The step gains what the profile shows the verify pass gaining (11.80 and 11.74 ms of routed gate/up).
- `gmix-q2` against `pmix-q2-fast` on this build and boot: +16.34 ms (+17.20 %) and +16.90 (+15.50 %) with the old
  kernels, **+4.68 (+4.93 %) and +5.12 (+4.70 %) with `2`**. (R125, on `eee615cdb`: +18.05 and +17.66.)
- On the first request at a new depth (pass 1) and on the sampler pass the step gains too (−11.93 ms at depth 0 on
  pass 1, −15.29 on the sampler pass, both gains). At 8,192 those two requests vary more between loads of one setting
  (up to 14.38 ms on pass 1 and 10.17 on the sampler pass), so their −14.30 and −6.10 ms fall within the spread
  (negligible by the rule).
- Tokens a step do not change with the switch (the same replies, the same accepted drafts: 2.44 and 2.46 tokens a
  step at depth 0 on passes 1 and 2, 2.64 at 8,192). `pmix-q2-fast` writes other replies and accepts more drafts on
  them (2.91 and 3.05 a step on pass 2), so its tok/s ([table2.md](table2.md)) is not a kernel comparison; the step is.

### Prompts

The 8,128-token prompt read at 143.3 and 148.9 tok/s unset, 139.5 with `1`, 146.7 and 147.5 with `2`, and 224.4 with
`pmix-q2-fast` ([table.md](table.md), [table2.md](table2.md)). The kernels serve products of 2 to 8 tokens only, so
long prompts keep `gmix-q2`'s gap (−34.3 % with `2`); that is registry row Q1872.

### What limits the new kernels

Each kernel's time a layer beside its static instruction count (SASS per 32 weights of one matrix at one member, from
the PTX notes; at four tokens almost every expert has one member) and beside the bytes it reads (est. at 38.84
distinct experts a layer, C827's estimate for random picks; real picks overlap more, so fewer bytes), depth 0
([limits.md](limits.md); [limits_d8192.md](limits_d8192.md) at 8,192):

| Type | setting | ms a layer | instructions | time per instruction, Q2_0 = 1 | est. GB/s | of the card's ~173 measured |
|---|---|---:|---:|---:|---:|---:|
| IQ2_XXS | old | 0.938 | 131.44 | 1.17 | 35.0 | 20 % |
| IQ2_XXS | CUDA C | 0.557 | 76.94 | 1.19 | 58.9 | 34 % |
| IQ2_XXS | PTX | 0.464 | 72.50 | 1.05 | 70.7 | 41 % |
| IQ2_S | old | 0.984 | 136.31 | 1.18 | 41.4 | 24 % |
| IQ2_S | CUDA C | 0.535 | 70.25 | 1.25 | 76.2 | 44 % |
| IQ2_S | PTX | 0.434 | 62.94 | 1.13 | 94.0 | 54 % |
| IQ2_XS | old | 1.038 | 142.69 | 1.19 | 35.4 | 20 % |
| IQ2_XS | CUDA C | 0.598 | 81.62 | 1.20 | 61.6 | 36 % |
| IQ2_XS | PTX | 0.496 | 73.00 | 1.11 | 74.2 | 43 % |
| IQ1_M | old | 0.549 | 76.69 | 1.17 | 50.7 | 29 % |
| IQ1_M | CUDA C | 0.331 | 52.50 | 1.03 | 84.0 | 49 % |
| IQ1_M | PTX | 0.268 | 45.69 | 0.96 | 103.8 | 60 % |
| Q2_0 | | 0.262 | 42.88 | 1.00 | 136.7 | 79 % |

- **The time follows the instruction count.** Across twelve kernel versions the time per instruction stays within
  0.96 to 1.25 times Q2_0's, while the counts range from 45.69 to 142.69. All four grid kernels make 4 random
  shared-memory lookups per 32 weights, where Q2_0 makes none. IQ1_M's 32 lanes share one table copy (the most bank
  conflicts; IQ2_S has 8 lanes a copy, IQ2_XXS and IQ2_XS 4), yet IQ1_M's PTX runs at 0.96 of Q2_0's time per
  instruction. So the lookups are not the limit, and no type's time orders by its lanes per copy.
- **Not the bytes:** the PTX kernels read at est. 41 to 60 % of the card's measured bandwidth, Q2_0's kernel at
  est. 79 %.
- **The PTX bought more time than its count:** from CUDA C to PTX the time fell 16.6 to 19.0 % for 5.8 to 13.0 % fewer
  instructions, so the PTX kernels also issue faster per instruction (1.03–1.25 → 0.96–1.13). The PTX notes list what
  differs besides the count: all row loads issued first, the table address in one instruction, I2F scales.
- **What is left over Q2_0's time with the PTX kernels:** IQ2_XXS +0.202 ms a layer (11 layers, +2.225 ms a pass),
  IQ2_S +0.172 (7, +1.203), IQ2_XS +0.234 (4, +0.935), IQ1_M +0.007 (2, +0.013): **est. 4.38 ms a pass.** Fewer
  instructions are what pays, so the parked count-cutting ideas reopen: the signs from a table for IQ2_XXS and IQ2_XS
  ([Q1879](../../../quants/experiments.md), static 72.5 → 56.5 and 73.0 → 57.0, est. 1.56 ms a pass at the PTX
  kernels' measured rate), IQ2_XS's high-half entries without their shift (Q1894) and IQ1_M's second selector without
  its shift (Q1898), beside the planned Q1888, Q1889 and Q1899.

## How it ran

- **Engine:** the fork branch `pascal-p106-v5-iq-mtok-fast` at `883f21f2c` (C1138's four kernels on `pascal-p106-v5`
  `2d839da1d`; `llama-server` 1.0.0-dev, build 13244). One build for every arm: the control is the same build with the
  switch unset. Nine P106-090 cards at stock clocks (no offsets applied that boot), the 4 GB host, nothing else
  running (the deep lane's quiet check before every load: no llama process, 0 MiB on every card). Every load ran in
  one boot of the rig.
- **Files:** `gmix-q2` (`Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-gmix-q2-0000{1,2,3}-of-00003.gguf`), shard 3
  the n-gram table; for `p0` `pmix-q2-fast` (`Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-0000{1,2,3}-of-00003.gguf`);
  the MTP draft `mtp-Qwen3.8-Flash-Next-draft.gguf`.
- **The line:** Flash-Next's v5 preset unchanged ([settings.env](../../../release/flash-next-v5/settings.env)), plus per
  arm only the profiler's variables and `GGML_CUDA_MTOK_IQ_FAST` ([run/c1146.sh](run/c1146.sh)). `p0` gives
  `pmix-q2-fast`'s first shard as the last `-m`, which llama-server takes (as in R125); each server log's
  `loading model` line names the file it read. A dry run checked that each arm's server line is `u0`'s with only its
  own variables added (`p0`'s: `u0`'s plus that `-m`). [loads.tsv](loads.tsv) and [loads2.tsv](loads2.tsv) hold every
  load's line as it ran.
- **Text:** R125's: the release check's recorded agent requests (`TAG='^t1swift/'`), prompts of exactly 64 and 8,192
  tokens, 256 tokens out at temperature 0, two passes per depth, then the sampler pass with a fixed seed
  (`SAMPLER_PASS=1`), four idle seconds before every request (`PAUSE=4`).
- **Loads**, in order. First lane, 09:33–10:44: `warm` (unset, untimed), `g0`, `g1`, `g2` (group profile, unset, `1`,
  `2`), `u0`, `u1`, `u2`, `u0b` (no profiler). Second lane, 10:49–11:05: `u2b` (a second load at `2`, for the class)
  and `p0` (`pmix-q2-fast`, unset). Loading took 343, 344 or 345 s with `gmix-q2` and 307 s with `pmix-q2-fast`. The
  loads tables' `model` column names the lane's `MODEL` (`gmix-q2`) for `p0` too; its server log names
  `pmix-q2-fast`.
- **The checks:** every `gmix-q2` load (the seven timed ones of the first lane and `u2b`) wrote the same greedy reply on
  both passes at each depth and the same sampler-pass reply, the same as R125's `gmix-q2` loads; `p0` wrote its own,
  the same on both passes. No graph reallocation in any request. No load logged that a PTX kernel failed to load (the
  loader falls back to the CUDA C kernel with a warning). The rig afterwards: no llama process, nine cards at 0 MiB, no
  NVIDIA Xid.
- **Tables:** on a CPU, standard-library Python ([run/tables.sh](run/tables.sh)): R121's `split_trace.py` cuts each
  group profile into one file per request, R49's `verify_parts.py` gives the verify pass by part and launch-group
  label, [run/types.py](run/types.py) the routed gate/up by type and setting against R125's load, R125's
  [products.py](../2026-11-25_r125_gmix_step/run/products.py) the settings product by product,
  [run/limits.py](run/limits.py) the kernels beside their instruction counts and bytes, [run/steps.py](run/steps.py)
  the step pair over both lanes with its classes, and the deep lane's `deep_table.py` makes [table.md](table.md) and
  [table2.md](table2.md).

## Files

| File | What it holds |
|---|---|
| [types.md](types.md), [types.tsv](types.tsv) | the routed gate/up by type: ms a layer and a pass at each setting and R125's, the verify pass by part; requests `d0p2`, `d8192p2`, `d0p1`, `d8192p1` |
| [limits.md](limits.md), [limits_d8192.md](limits_d8192.md), [limits.tsv](limits.tsv) | each kernel's time a layer beside its static instruction count and est. bytes, and what is left over Q2_0's time |
| [steps.md](steps.md), [steps.tsv](steps.tsv) | ms a step of every unprofiled load at each depth and pass, means, spreads, classes against `u0`; the replies' identity |
| [products-g0-g1.md](products-g0-g1.md), [products-g0-g2.md](products-g0-g2.md), [products-g1-g2.md](products-g1-g2.md) and their `.tsv` | the verify pass by part, by product and by label, setting against setting |
| [verify/](verify/) | `verify_parts.py`'s tables per load and request (`g0`, `g1`, `g2`; `d0p1` … `d8192s`): `parts.tsv`, `labels.tsv`, `per_card.tsv`, `accounting.tsv` |
| [table.md](table.md), [lane_classes.tsv](lane_classes.tsv) | the first lane's own tables: tok/s, the step on each pass, prompts, the draft, the sampler pass, the checks (control `u0`) |
| [table2.md](table2.md), [lane_classes2.tsv](lane_classes2.tsv) | the same for the second lane (`u2b` the control, `p0`) |
| [loads.tsv](loads.tsv), [loads2.tsv](loads2.tsv), [requests.csv](requests.csv), [requests2.csv](requests2.csv) | per load its line and binary as it ran; per request its counts, times and reply hashes (first and second lane) |
| [run/c1146.sh](run/c1146.sh) | the run through `bench/deep_lane/remote.sh` (dry run with its check, start, wait, collect; `PMIX` and `LOADS` for the second lane) |
| [run/tables.sh](run/tables.sh), [run/types.py](run/types.py), [run/limits.py](run/limits.py), [run/steps.py](run/steps.py) | the tables from the two lanes' results folders |

The group-profile traces (95 MB a load) are not published; the tables above are what the tools read from them.

## Commands

```bash
# from a machine that reaches the rig over ssh, at the repository root (paths are the rig's)
export RIG BIN=<llama-server of 883f21f2c> MODEL=<gmix-q2 shard 1> MODEL_DRAFT=<mtp-Qwen3.8-Flash-Next-draft.gguf> \
  TEXT_LOCAL=<proxy.jsonl.bodies.gz here>
D=docs/data/2026-11-26_c1146_iq_kernels_rig1/run
WORK=/tmp/c1146 STEP=dry bash $D/c1146.sh && WORK=/tmp/c1146 STEP=start bash $D/c1146.sh
WORK=/tmp/c1146 STEP=wait bash $D/c1146.sh && WORK=/tmp/c1146 STEP=collect OUT=<first lane here> bash $D/c1146.sh
# the second lane: a second load at 2 and pmix-q2-fast
export PMIX=<pmix-q2-fast shard 1> LOADS="u2b:u2 p0:p0"
WORK=/tmp/c1146b STEP=dry bash $D/c1146.sh && WORK=/tmp/c1146b STEP=start bash $D/c1146.sh
WORK=/tmp/c1146b STEP=wait bash $D/c1146.sh && WORK=/tmp/c1146b STEP=collect OUT=<second lane here> bash $D/c1146.sh
# the tables, on any machine with Python 3
S=$PWD OUT=<first lane here> OUT2=<second lane here> RES=<tables> bash $D/tables.sh
```

Provenance and licence: measurements on my own rig; the recorded agent text is the release check's
([2026-11-05_release_flashnext](../2026-11-05_release_flashnext/README.md)); the model files are derived from
UkisAI's Swift-1.5 GSQ-RCO files under their licence.
