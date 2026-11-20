# R53 on the v5 engine: what Flash-Next's draft costs in prompt reading (Q1215; 2026-11-20)

**On the v5 engine and preset the MTP draft makes prompt reading 3.8 to 3.9 % slower on long chunks after a load's
first, and 13.9 % slower on its first. On v3.0's line it was 28.4 to 41.3 %.** The draft model reads every prompt
token too, so a prompt costs more with it loaded. Here the v5 preset as it is (with the draft) runs against the same
line without the draft, two loads each, on prompts of 8,192, 16,384 and 32,768 tokens:

| Chunk | With the draft, tok/s (ms a prompt token) | Without it | The draft adds, ms a prompt token | tok/s | v3.0's line ([Q1215](../../../quants/experiments.md)) |
|---|---|---|---:|---:|---|
| 8,068 tokens on an empty cache, the load's first prompt | 193.5 (5.168), 193.6 (5.166) | 226.5 (4.414), 222.8 (4.488) | +0.716 | −13.9 % | 8,128 on an empty cache: +2.23 ms, −28.4 % |
| 16,260 tokens on an empty cache | 319.5 (3.130), 309.9 (3.227) | 327.8 (3.051), 326.6 (3.062) | +0.122 | −3.8 % | (no such chunk) |
| 16,900 tokens on 15,804 cached | 225.9 (4.426), 221.1 (4.523) | 232.6 (4.298), 232.5 (4.301) | +0.175 | −3.9 % | 16,388 on 16,316 cached: +4.28 ms, −38.8 % |

- **The draft's cost per prompt token is 68 % lower than on v3.0's line on an empty cache (0.716 against 2.23 ms) and
  96 % lower on about 16,000 cached tokens (0.175 against 4.28 ms).** It no longer grows with depth: 0.122 ms on a
  fresh 16,260-token chunk and 0.175 ms on 16,900 tokens over 15,804 cached. On v3.0's line it rose from 2.23 to 5.89
  ms between an empty cache and 32,188 cached tokens.
- **Over a whole load** (the three chunks, 41,228 tokens) the line reads 246.3 and 241.7 tok/s with the draft (167.39
  and 170.60 s) and 261.2 and 259.8 without it (157.86 and 158.68 s): −6.3 %, 10.73 s a load. On v3.0's line the draft
  added 217.6 s to a load's 50,124 prompt tokens.
- The lane classes the line without the draft a gain in prompt reading at every depth: the draft is still a cost, just
  a small one. The two arms' loads repeat closely: within 3.7 tok/s without the draft and within 9.6 tok/s with it.
- The changes since v3.0's run that act on this: the draft's micro-batch of 512 instead of 128 (R53's first run took
  0.37 to 0.45 ms off a prompt token with it), its catch-up writing only the draft layer's caches
  (`LLAMA_MTP_KV_ONLY=1`), the catch-up of one prompt call running while the next call starts
  (`LLAMA_MTP_CATCHUP_OVERLAP=1`), and prompt micro-batches on the sparse attention, which takes the draft's attention
  too.

## Where the draft's time sits: the end of each chunk

The server prints a progress line after every 4,096 tokens of a prompt. Almost all of the draft's extra time falls
after a chunk's last progress line, in the last call, which has no next call for the draft's catch-up to overlap
(est. the reason; the mean of each arm's two loads, s):

| Chunk | The draft adds, whole chunk | before the last progress line | after it |
|---|---:|---:|---:|
| 8,068 on an empty cache | 5.77 | 1.00 | 4.78 |
| 16,260 on an empty cache | 1.98 | 0.07 | 1.91 |
| 16,900 on 15,804 cached | 2.96 | 1.67 | 1.29 |

So a reply's first token waits on the draft's last catch-up, which is what code row C853 (Q1524) would move behind the
first token. Its worth here is at most the "after" column: est. up to 4.78 s of a fresh 8,068-token prompt's 41.69 s
(11.5 %), 1.91 s of the 16,260-token chunk's 51.68 s (3.7 %) and 1.29 s of the 32,768-token chunk's 75.62 s (1.7 %).
The first chunk's larger cost is est. a one-time cost of the load's first long prompt in the draft. Over a load the
whole draft cost is 6.3 % of prompt reading, against the 30.3 % on v3.0's line that the
[prompt-path review](../../research/notes/v5_review_prompt_path_2026-11.md) priced C853 by (row 8 there: est. 6 % of a
task then; est. about 1 % now).

Without the draft the time after the last progress line is long too: 23.44 and 23.87 s for the 8,068-token chunk's last
3,972 tokens (its first 4,096 took 12.18 and 12.34 s) and 13.89 and 13.85 s after the 16,900-token chunk's last line,
which leaves 516 tokens. Est. this is the checkpoint stop that code row C907 (Q1644) found on Ornith; its line in the
registry has these numbers.

## What was run

- **Rig and settings:** rig 2, nine P106-090 cards at stock (no clock offsets, the 75 W default power limit), freshly
  booted about two minutes before the run. Nothing else ran on the rig during the lane.
- **Model and preset:** Qwen3.8-Flash-Next as [release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env)
  runs it on this day: UkisAI's Swift-1.5 GSQ-RCO Q2_0 (`Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{1,2,3}-of-00003.gguf`;
  the third shard is its n-gram table) and the MTP draft (`mtp-Qwen3.8-Flash-Next-draft.gguf`), three tokens a step,
  context 65,536, with the sparse-attention ratio of 0.75 and the prompt taper `short=1400:64`.
- **Build:** fork `pascal-p106-v5` at `602691153`, the candidate `pascal-p106-v5-cand-b12483` (upstream b12483
  merged). Its `llama-server` md5 is `86607661d5a19d1a90d19ee4db10c90b`.
- **Arms:**
  - `draft`: the preset as it is (control).
  - `nodraft`: the same line without `-md` and without the draft's flags (`--spec-type`, `--spec-draft-n-max`,
    `--spec-draft-n-ubatch`, `--spec-draft-sampling`, `--spec-draft-p-min`). Every environment switch is kept,
    including the draft's, which then have nothing to act on.
- **The lane** ([bench/deep_lane](../../../bench/deep_lane/README.md)): a warm-up load of `draft` at depth 0, then the
  loads `d1` (draft), `n1`, `n2` (no draft) and `d2` (draft). Each load primes the cache at 8,192, 16,384 and 32,768
  tokens (each prime reads a chunk on top of what the cache holds and writes one token), then writes 16 tokens at
  temperature 0 twice per depth. The prompts are cut from this project's recorded agent text.
- **The chunks:** the 16,384-token prime found no checkpoint inside the first chunk's shared prefix and read its
  prompt from the 60-token checkpoint in every load, so the three chunks are 8,068 tokens on an empty cache, 16,260 on
  an empty cache and 16,900 on 15,804 cached. v3.0's run read its second chunk on 7,612 cached tokens, so only its
  first and third chunks match these.
- **Time:** the warm-up 20:48:28, `d1` 20:52:26, `n1` 20:59:22, `n2` 21:05:46, `d2` 21:12:10, the end 21:19:05.
  Each load read the files in 224 or 225 s with the draft and 208 s without.

## The rest of the lane (`table.md`)

- **Generation is not this run's measure.** 16 tokens a request show the draft's step but not its worth: the draft
  accepted 7 of 23 drafted tokens at 8,192, 9 of 16 at 16,384 and 8 of 16 at 32,768 on the first request. Over all
  depths the lane gives 16.35 and 16.85 tok/s with the draft against 17.58 and 17.58 without. The draft's worth in
  generation is measured on 256-token requests (Q1153: not a loss at any depth up to 49,152 tokens) and on the
  agent check.
- **Replies:** at 16,384 every greedy reply is the same in all four loads. At 8,192 and 32,768 the line without the
  draft writes another greedy reply than the line with it; each arm's two loads agree.
- **Memory** (`memory.tsv`): with the draft card 8 holds 5,545 MiB and without it 2,359; the other cards hold 26 to
  46 MiB more with it, 3,500 MiB over the nine cards. The host's lowest MemAvailable during a load was 333 and
  344 MiB with the draft and 696 and 698 without it. The RAM watchdog stopped no load.
- **Graph reallocations:** none in any load's 10 requests. Every load started on a quiet rig. The hottest card
  reached 46 °C.

## Files

| File | What |
|---|---|
| `cost.md` | the draft's cost: each prime by load (prompt tok/s, ms a prompt token, wall time), per depth the mean ms a prompt token with and without the draft, and the whole load (`run/draft_cost.py`) |
| `primes.tsv` | the same per prime as a table: load, arm, depth, tokens read and cached, prompt and wall ms, tok/s, ms a prompt token |
| `table.md` | the lane's tables as `deep_table.py` prints them, `draft` as the control, classes on the load |
| `classes.tsv` | `nodraft` against `draft` per metric and depth: each load's value, spread, delta, percent, class |
| `requests.csv` | one line per request (`deep_client.py`): prompt and generated tokens, times, draft counters, the reply's md5s |
| `loads.tsv` | one line per load: arm, start time, seconds to load, its variables and the server line |
| `memory.tsv` | per load: MiB on each card, MemAvailable when the model was up, the lowest MemAvailable during the load |
| `cards.md` | the card log over the whole lane ([scripts/card_table.py](../../../scripts/card_table.py)) |
| `run/r53.sh` | the run script: R69's deep-lane design with the draft as the arm |
| `run/draft_cost.py` | makes `cost.md` and `primes.tsv` from the lane's `requests.csv` |

The progress lines and their times (the table "Where the draft's time sits") are from the server logs, which stay out
of this repository: `prompt processing, n_tokens = …, t = … s` per 4,096 tokens and `prompt eval time` at a chunk's
end.

## Commands

From the repository's root. The variables:
- `RIG`: the rig's ssh alias.
- `WORK`: the run folder on the rig.
- `BIN`: the build's `llama-server` there.
- `MODEL`: the first shard of the Swift GSQ-RCO Q2_0 file on the rig, with the other two beside it.
- `MODEL_DRAFT`: the MTP draft on the rig.
- `TEXT_LOCAL`: the recorded agent text, the release check's `proxy.jsonl.bodies.gz`, as in R69.

```bash
export RIG=<rig> WORK=<folder on the rig> \
  BIN=/opt/llama-builds/llama.cpp-20261117-602691153-pascal-p106-v5-cand-b12483/bin/llama-server \
  MODEL=<models>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  MODEL_DRAFT=<models>/mtp-Qwen3.8-Flash-Next-draft.gguf TEXT_LOCAL=<proxy.jsonl.bodies.gz>
R=docs/data/2026-11-20_r53_flash_draft_prompt/run/r53.sh
STEP=dry bash $R      # the two server lines, nothing started
STEP=start bash $R    # the lane, detached on the rig
STEP=wait bash $R     # returns when the lane has ended (here 31 min)
OUT=<folder> STEP=collect bash $R
# the tables, on any machine with Python 3 (here the machine that drove the run):
D=docs/data/2026-11-20_r53_flash_draft_prompt
python3 -I $D/run/draft_cost.py <folder> --tsv $D/primes.tsv > $D/cost.md
python3 -I bench/deep_lane/deep_table.py <folder> --control draft --out $D/table.md --tsv $D/classes.tsv
python3 -I docs/data/2026-11-18_c865b_ornith_glu_rig1/run/memory_table.py <folder> $D/memory.tsv
python3 -I scripts/card_table.py <folder>/cards.csv --out $D/cards.md
cp <folder>/requests.csv <folder>/loads.tsv $D/
```

**Provenance:**
- The model file is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next with its n-gram table cut into a third
  shard, and the MTP draft, as [release/flash-next-v4](../../../release/flash-next-v4/README.md) describes them.
- The recorded agent text is this project's own release check.
- v3.0's numbers are R53's first run, the
  [fourth run in 2026-11-10_deep_lane](../2026-11-10_deep_lane/README.md#the-fourth-run-the-drafts-micro-batch-in-prompt-reading-q1215)
  and its second pair.
- The tables carry the licence of this repository.
