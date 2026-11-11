# R76 (Q1278): C620's KV-only MTP catch-up on a P106 (2026-11-11)

What this measured: whether `LLAMA_MTP_KV_ONLY=1` (C620, [the change](../2026-11-11_c620_draft_kv_catchup/README.md))
leaves the MTP draft's guesses unchanged on a P106 rig, and how much of the draft's cost in prompt reading it takes away
at depth. The switch cuts every draft pass that outputs nothing (the catch-up over a prompt chunk, or over a verify
step's accepted tokens) down to the writes of the draft layer's caches. The registry row is Q1278 in
[quants/experiments.md](../../../quants/experiments.md).

The build is Miner Llama's v4 night merge, `pascal-p106-v4-b12246-night` at `c408dabcb`
([2026-11-11_v4_night_merge](../2026-11-11_v4_night_merge/README.md)), which holds C620's change. No other switch it adds
was set, so with this switch off it computes what `pascal-p106-v4` computes. The model and line are R53's: UkisAI's
Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next on nine P106-090 cards, the v3 preset (`-b 2048 -ub 512`, the MTP draft
at three tokens, catch-up passes of 128 rows) with the sparse Pascal attention on (`GGML_CUDA_FATTN_SPARSE_PASCAL=1`),
greedy, 16 tokens a request.

## Result

- **The drafts are the same.** In the identity lane (depths 0 and 32,768, three loads in the order draft, switch,
  draft), the switch's load gives the same 39 verify steps as the two control loads, every field the same (the drafted
  tokens, their candidates and probabilities, the main model's targets and probabilities, the tokens kept). The two
  control loads agree with each other on all 39 ([lane/dump_compare.tsv](lane/dump_compare.tsv)). The greedy replies
  are the same in every load. In the prompt run the greedy reply with the switch is the control's at every depth, and
  every draft load accepts the same tokens (14 of 46, 14 of 44, 18 of 36, 16 of 32 and 20 of 28 at the five depths).
- **Prompt reading is a gain at every depth from 8,192 tokens on: +10.0 % over a load's four prompt chunks.** The
  chunks (50,124 tokens) read at 101.4 and 101.8 tok/s with the switch, against 92.2 and 92.6 without it and 145.4 and
  145.2 with no draft; they took 494.4 and 492.5 s, against 543.8 and 541.5 and 344.7 and 345.1. The draft adds 2.96 ms
  a prompt token with the switch against 3.95 without it: **the switch takes away 24.9 % of the draft's prompt cost.**

  | Depth (tokens) | Read, on cached | Draft, tok/s | Draft + switch, tok/s | No draft, tok/s | Switch against draft | Taken away |
  |---:|---|---|---|---|---|---:|
  | 0 | 64 on 0 | 33.6, 32.8 | 33.5, 32.9 | 34.1, 33.7 | −0.0 %, negligible | |
  | 8,192 | 8,128 on 0 | 138.1, 138.6 | 146.4, 147.9 | 180.1, 181.4 | +6.4 %, gain | 25.5 % |
  | 16,384 | 8,708 on 7,612 | 114.8, 115.1 | 122.9, 123.6 | 167.9, 168.1 | +7.2 %, gain | 21.3 % |
  | 32,768 | 16,388 on 16,316 | 93.6, 94.4 | 103.2, 103.7 | 150.8, 150.7 | +10.0 %, gain | 24.2 % |
  | 49,152 | 16,900 on 32,188 | 72.2, 72.3 | 80.8, 80.8 | 121.6, 120.9 | +11.8 %, gain | 26.2 % |

  Each value is one load ([prompt/draft_ubatch.md](prompt/draft_ubatch.md)). The largest spread between two loads of
  one arm at 8,192 tokens and beyond is 1.5 tok/s.
- **Most of the part that grows with depth is left.** The draft adds 1.70, 2.75, 4.00 and 5.59 ms a prompt token at
  8,192, 16,384, 32,768 and 49,152 tokens, and 1.26, 2.16, 3.03 and 4.13 with the switch: of the 3.89 ms it grows by,
  2.87 remain. The cut pass skips the draft layer's attention over its cache, so most of the growing part lies outside
  that attention.
- **The decode step is faster too, since each verify step's catch-up is cut as well.** On the repeated request at
  each depth (pass 2) a step takes 109.54 and 109.56 ms with the switch against 111.12 and 111.24 at 0 tokens
  (−1.5 %, gain), 241.01 and 241.00 against 243.90 and 243.77 at 8,192 (−1.2 %, gain), 129.67 and 129.50 against
  144.60 and 132.04 at 16,384 (−6.3 %, negligible: the control's spread is 12.57 ms), 132.94 and 133.52 against 138.74
  and 135.65 at 32,768 (−2.9 %, gain) and 152.25 and 151.85 against 154.48 and 154.22 at 49,152 (−1.5 %, gain)
  ([prompt/classes.tsv](prompt/classes.tsv)). The first request after each prompt chunk (pass 1) varies from load to
  load in both arms (232.01 and 158.11 ms a step without the switch at 32,768; 261.01 and 156.89 with it at 49,152), so
  only 16,384 has a class there: 139.31 and 138.61 against 208.88 and 211.64 ms, a gain.
- **The 4 GB host held.** The RAM watchdog (stop below 60 MiB available) never acted. Lowest MemAvailable per load:
  154 and 249 MiB without the switch, 93 and 66 with it, 665 and 548 with no draft
  ([prompt/host_memory.tsv](prompt/host_memory.tsv)). The lowest values follow what the kernel swapped: `kvonly2` never
  swapped (55 MiB of swap in use at most, none moved during its prompts), while the control loads moved 691 and 812 MiB
  to swap during theirs. The cards hold the same memory with the switch (44,101 MiB in all after loading).
- **The long prime in the identity lane (one load each, a diagnostic):** the 32,704-token prompt on an empty cache took
  272,058.646 ms with the switch against 292,653.731 and 293,275.454 without it (120.2 against 111.7 and 111.5 tok/s).

The switch therefore joins the v4 preset candidate (Q1254) for the agent check of the stacked line (Q1261). The night
build logs no line of its own for this switch; each load's environment is in its `loads.tsv` line.

## How it was measured

1. **Identity, untimed** (`bench/deep_lane`): three loads, `draft1` (control), `kvonly1` (`LLAMA_MTP_KV_ONLY=1`) and
   `draft2` (the control again, as its own arm so that it writes its own dump), each with `LLAMA_SPEC_DUMP` (every
   verify step), at depths 0 and 32,768, the RAM watchdog at `MIN_AVAIL_MB=60`. `deep_table.py`'s checks compare the
   greedy replies; R82's [dump_compare.py](../2026-11-11_r82_c702_p106/run/dump_compare.py) pairs the three dumps step
   by step without the fields that name the process (`run`, `step`, `task`).
2. **The prompt table** (R53's command, [its run](../2026-11-10_deep_lane/README.md)): a warm-up load of the draft,
   then two rounds of `draft`, `kvonly` and `nodraft` (the same line without the draft), depths 0, 8,192, 16,384, 32,768
   and 49,152. Each depth's prime reads a chunk on top of the previous depth's cache, then two decode requests of 16
   tokens follow (pass 1 the first at the new depth, pass 2 the same request again).
3. **The tables**, on a build machine's CPU: `deep_table.py` (generation, the step, prompt reading, the checks; the
   rig's own `table.md` and `classes.tsv` are the same bytes), R53's
   [draft_ubatch.py](../2026-11-10_deep_lane/run/draft_ubatch.py) (prompt reading with the switch as a setting of the
   draft line against the control and the floor, the share of the draft's cost taken away, the host's memory) and
   [host_memory.py](../2026-11-10_deep_lane/run/host_memory.py).

The class follows the project's rule: a gain when the arms' means differ by more than the larger of the two arms'
spreads (the difference between an arm's fastest and slowest load), a loss when the other way, else negligible.

## Files

| File | What |
|---|---|
| [lane/table.md](lane/table.md), [lane/classes.tsv](lane/classes.tsv), [lane/requests.csv](lane/requests.csv), [lane/loads.tsv](lane/loads.tsv) | the identity lane: `deep_table.py`'s tables and checks, every request, each load's line |
| [lane/dump_compare.tsv](lane/dump_compare.tsv) | the 39 verify steps: whether the control loads agree and the switch's load is the same, and which fields differ |
| [lane/host_memory.tsv](lane/host_memory.tsv) | the host's memory per load and request in the lane |
| [prompt/table.md](prompt/table.md), [prompt/classes.tsv](prompt/classes.tsv), [prompt/requests.csv](prompt/requests.csv), [prompt/loads.tsv](prompt/loads.tsv) | the prompt run: `deep_table.py`'s tables and checks, every request, each load's line |
| [prompt/draft_ubatch.md](prompt/draft_ubatch.md), `prompt/draft_ubatch_prompt.tsv`, `prompt/draft_ubatch_memory.tsv`, `prompt/draft_ubatch_generation.tsv` | `draft_ubatch.py`: prompt reading per depth with the share taken away, the host's and the cards' memory per load, generation as a check |
| [prompt/host_memory.tsv](prompt/host_memory.tsv) | the host's memory per load and request in the prompt run |
| [run/r76_run.sh](run/r76_run.sh) | the run: both parts, their wait and collection |

## Commands

From a machine that only runs ssh, at the repository's root, with `RIG`, `WORK`, `BUILDS`,
`BUILD=llama.cpp-20261111-c408dabcb-pascal-p106-v4-b12246-night`, `MODELS` and `RECORDED` set (the variables are in the
script's header):

```bash
STEP=lane bash docs/data/2026-11-11_r76_c620_p106/run/r76_run.sh
STEP=wait PART=lane bash docs/data/2026-11-11_r76_c620_p106/run/r76_run.sh
STEP=collect PART=lane OUT=<here>/lane bash docs/data/2026-11-11_r76_c620_p106/run/r76_run.sh
STEP=prompt bash docs/data/2026-11-11_r76_c620_p106/run/r76_run.sh
STEP=wait PART=prompt bash docs/data/2026-11-11_r76_c620_p106/run/r76_run.sh
STEP=collect PART=prompt OUT=<here>/prompt bash docs/data/2026-11-11_r76_c620_p106/run/r76_run.sh
```

Then on a build machine's CPU, in the folder that holds `lane/` and `prompt/`, with `R` the repository:

```bash
python3 -I $R/docs/data/2026-11-11_r82_c702_p106/run/dump_compare.py --control lane/dump-draft.jsonl \
  --control2 lane/dump-draftb.jsonl --arm lane/dump-kvonly.jsonl --out lane/dump_compare.tsv
for p in lane prompt; do
  python3 -I $R/bench/deep_lane/deep_table.py $p --out $p/table.md --tsv $p/classes.tsv
  python3 -I $R/docs/data/2026-11-10_deep_lane/run/host_memory.py $p > $p/host_memory.tsv
done
python3 -I $R/docs/data/2026-11-10_deep_lane/run/draft_ubatch.py prompt --control draft --floor nodraft \
  --out prompt/draft_ubatch.md --tsv prompt/draft_ubatch
```

The lane ran 19:27:41 to 19:54:36 (about nine minutes a load), the prompt run 19:55:11 to 21:12:28 (about 13 minutes a
load with the draft, 10 without it). Every load started on a quiet rig. The prompts are built from the release agent
check's recorded traffic (`TAG='^t1swift/'`, [bench/deep_lane](../../../bench/deep_lane/README.md)); only their md5
is published here.
