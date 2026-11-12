# C703's kept catch-up on a P106: the drafts' identity (R84 a, Q1304, 2026-11-12)

What this is: the identity step of C703's rig run ([the change, its CPU check and the rig design](../2026-11-11_c703_catchup_overlap/README.md#on-the-rig),
registry [Q1304](../../../quants/experiments.md)). With `LLAMA_MTP_CATCHUP_OVERLAP=1`, llama-server holds the MTP
draft's catch-up over a prompt call until the target's next call is queued. The catch-up then runs on cards 4 and 8
while that call's first micro-batches cross cards 0 to 7. On the CPU the switch leaves the draft's caches, its drafts
and the server's checkpoints the same bytes. On the cards the draft's passes run beside the target's next call, so the
identity is checked again here. The step is untimed: it gates the overlap rung of the prompt ladder (R123a), which
times the switch.

One deep-lane run ([bench/deep_lane](../../../bench/deep_lane/README.md)) on nine P106-090 cards with Miner Llama's
`pascal-p106-v4` at `d097d6d88`, which holds C703's `db253c8a7` through the night merge `5d929bf0a`. No other switch
of the v4 line was set. Three loads, in this order: the draft line (`draft1`), the same line with the switch
(`overlap1`) and the draft line again (`draft2`). Each load wrote every verify step with `LLAMA_SPEC_DUMP`, at depths 0
and 32,768 tokens.

## Result

**The identity holds on a P106: the switch's load drafts the same tokens with the same probabilities as both control
loads at every verify step.**

- **Every verify step the same.** Each load wrote 39 verify steps, with 113 drafted tokens in all. `draft1` and `draft2`
  agree on all 39 steps, and `overlap1` gives the same 39 ([dump_compare.tsv](dump_compare.tsv),
  [dump_compare.txt](dump_compare.txt)). This holds for the drafted tokens and the draft's candidates, probabilities
  and picks at each drafted position. It also holds for every other field of the step: the main model's targets and
  their probabilities, the tokens kept, the position. The largest difference between two loads' draft probabilities
  is 0.00000.
- **The same replies and acceptance.** All three loads wrote the same greedy reply (token ids' md5) on both passes at
  each depth: `9da71beb…` at 0, `1ca520b0…` (pass 1) and `0795e350…` (pass 2) at 32,768. They also accepted the same
  draft tokens: 7 of 23 at 0, 8 of 18 and 8 of 16 at 32,768 ([requests.csv](requests.csv)).
- **The switch was on.** The `overlap1` server log lists it among the switches set at load (C770). It also has the
  switch's own line, `LLAMA_MTP_CATCHUP_OVERLAP: a prompt call's catch-up runs once the target has the next call
  queued`, and at exit `LLAMA_MTP_CATCHUP_OVERLAP: 20 catch-ups ran after the next call was queued`. Neither control
  log has either line ([log_lines.tsv](log_lines.tsv)).
- **Card 8 holds 80 MiB more with the switch,** as estimated (80.0 MiB, the second half of the target's h rows): 4,727
  against 4,647 MiB after loading, and 4,787 against 4,707 at the most. The other eight cards hold the same in all three
  loads ([cards_max.tsv](cards_max.tsv)). The host's lowest MemAvailable per load: 462 MiB (`draft1`), 217
  (`overlap1`) and 191 (`draft2`); the RAM watchdog (floor 60 MiB) never acted ([host_memory.tsv](host_memory.tsv)).

**So the overlap rung of R123a can run:** the switch is exact on the cards.

### Seen on the way (one load with the switch, not classed)

The run was not designed to time the switch: it has only one load with it. Every load did start on a quiet machine,
so its times are valid measurements. The 32,768-token prime (32,704 tokens read on an empty cache) took 207,879.268 ms
with the switch, against 293,547.329 and 293,288.089 ms without it (157.3 against 111.4 and 111.5 tok/s). That is
−85,538.4 ms, −29.15 % against the controls' mean, whose two loads are 259.2 ms apart. It is larger than C703's
estimate for a 4,097-token prompt (up to 2.6 s, 8.7 %). At this depth each catch-up is longer: R76 measured the draft
adding 4.00 ms a prompt token at 32,768 tokens, and this load ran 20 catch-ups behind the next call. The class needs
repeated loads, so the timing is R123a's overlap rung (R84 b and c). The decode steps are not classed either: pass 1
at 32,768 is 435.22 and 310.08 ms in the two control loads, a spread wider than any switch's effect.

## What ran

One deep-lane run on one rig, nothing else on the rig during the loads, 13:55:09 to 14:21:11 (26.0 minutes). Each
server came up in 225, 225 and 227 s.

| Load | Arm | Start | Its own variables |
|---|---|---|---|
| `draft1` | `draft` | 13:55:09 | `LLAMA_SPEC_DUMP=$WORK/dump-draft.jsonl` |
| `overlap1` | `overlap` | 14:04:22 | `LLAMA_MTP_CATCHUP_OVERLAP=1 LLAMA_SPEC_DUMP=$WORK/dump-overlap.jsonl` |
| `draft2` | `draft` | 14:12:03 | `LLAMA_SPEC_DUMP=$WORK/dump-draft.jsonl` |

- **The line:** [the v3 preset](../../../release/flash-next-v3/settings.env) with `GGML_CUDA_FATTN_SPARSE_PASCAL=1`
  (R53's line, as R76 and R91 ran it): UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next with the MTP draft at
  three tokens, `-b 2048 -ub 512`, catch-up passes of 128 rows, the RAM watchdog at `MIN_AVAIL_MB=60`.
- **Each load:** the client's warm-up, then at 0 and 32,768 tokens a prime and two decodes of 16 greedy tokens. The
  prompts are cut from the release check's recorded agent requests (`TAG='^t1swift/'`); every load and pass got the
  same token ids.
- **The dumps:** both `draft` loads append to one file. Each server start writes its own `run` (its start time), so
  [run/r84a_compare.py](run/r84a_compare.py) splits them by `run`. It then drops the fields that name the process
  (`run`, `step`, `task`) and pairs the steps in file order, as R82's and R76's comparisons do. The dumps (68,522 and
  34,261 bytes) hold token ids of the recorded prompts and stay out of this folder.
- **The build:** `pascal-p106-v4` `d097d6d88` (llama-server `build 12661`), the rig tarball of C796's kernel merge
  (md5 `37a4c3731f76455dce14013210b2fac7`), in place of C703's own tarball: it holds C703's code.
- **The tables ran on a build machine's CPU** (`nice -n 19`), never on the rig. Rerun there, the deep lane's own tables
  ([table.md](table.md)) are the same bytes as the rig's.

## Files

| File | What it holds |
|---|---|
| [dump_compare.tsv](dump_compare.tsv) | per verify step: position, prompt size, drafted tokens, accepted; whether the controls agree and `overlap1` is the same, on the draft's fields and on every field; the fields that differ |
| [dump_compare.txt](dump_compare.txt) | the comparison's totals and verdict |
| [log_lines.tsv](log_lines.tsv) | per load: the switches set at load and the switch's own lines, with their line numbers |
| [cards_max.tsv](cards_max.tsv) | per load and card: the most MiB used over the load (from the deep lane's 10 s card log) |
| [host_memory.tsv](host_memory.tsv) | the host's memory per load and request from the RAM watchdog ([host_memory.py](../2026-11-10_deep_lane/run/host_memory.py)) |
| [table.md](table.md), [classes.tsv](classes.tsv) | the deep lane's own tables and checks (control `draft`; one load with the switch, so no class) |
| [requests.csv](requests.csv), [loads.tsv](loads.tsv) | the deep lane's line per request and per load (the binary written as `$BIN/llama-server`, the work folder as `$WORK`) |
| [run/r84a_run.sh](run/r84a_run.sh) | the run: start, wait and collect (with the two dumps) through `bench/deep_lane/remote.sh` |
| [run/r84a_compare.py](run/r84a_compare.py) | the dump comparison |

## Commands

```bash
# from a machine that reaches the rig over ssh, at this repository's root (RIG: its ssh alias; BIN: the build's bin/
# on the rig; MODEL, MODEL_DRAFT: paths on the rig; TEXT_LOCAL: the release check's recorded requests,
# proxy.jsonl.bodies.gz; OUT: a results folder here)
D=docs/data/2026-11-12_r84a_c703_identity/run
RIG=$RIG BIN=$BIN MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT TEXT_LOCAL=$TEXT_LOCAL STEP=start bash $D/r84a_run.sh
RIG=$RIG STEP=wait bash $D/r84a_run.sh
RIG=$RIG OUT=$OUT STEP=collect bash $D/r84a_run.sh
ssh $RIG 'rm -rf /tmp/deep-lane'      # after checking the copies

# on a build machine, CPU only, from this repository's root (OUT: the collected folder; RES: the tables)
nice -n 19 python3 -I $D/r84a_compare.py --draft $OUT/dump-draft.jsonl --overlap $OUT/dump-overlap.jsonl \
  --out $RES/dump_compare.tsv > $RES/dump_compare.txt
nice -n 19 python3 -I bench/deep_lane/deep_table.py $OUT | cmp - $OUT/table.md   # the rig's own tables
nice -n 19 python3 -I docs/data/2026-11-10_deep_lane/run/host_memory.py $OUT > $RES/host_memory.tsv
grep -a -n -E 'LLAMA_MTP_CATCHUP_OVERLAP: |common_params_print_info: switches: ' $OUT/server-*.log
```

## Limits

- **One load with the switch.** The identity needs only one: every step is compared where the two control loads agree,
  and they agree on all 39. The prime's time above is one load against two and has no class.
- **Two depths, 16 tokens a request.** The steps cover a short prompt, a 32,704-token prompt (the load kept 20
  catch-ups in all), and the decodes after each. Longer generations and other prompt sizes are R123a's replay.

## Provenance and licence

Measured on 2026-11-12 on one rig of nine P106-090 cards, with the Miner Llama fork's `pascal-p106-v4` (llama.cpp,
MIT). The tables are this project's own measurements. The models are UkisAI's Swift-1.5 GSQ-RCO Q2_0 of
Qwen3.8-Flash-Next and the MTP draft; neither is republished. The prompts are cut from this project's recorded agent
requests; they, the replies and the dumps are not part of this folder (`requests.csv` and the tables hold only
checksums and counts).
