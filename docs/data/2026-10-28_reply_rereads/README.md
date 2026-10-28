# Reply re-reads on the agent check, and the prompt path's per-request costs (Q260, Q261, Q263), 2026-10-28

What makes an agent turn wait before its first token on the v2.0 release line (Flint Q8_0, nine P106-090 cards,
`-c 262144 -np 1 -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 -bs --ctx-checkpoints 4 --cache-ram 0 --pipe-depth 4`), measured
offline from the agent check's own logs, and three engine switches that address it. Result pages: registry rows
[Q260, Q261, Q263](../../../quants/experiments.md) and the [Flint plan](../../research/notes/flint_v3_plan.md).

## Results

**1. Where the prompt cache stops after a reply.** In 3 release-line runs of the agent check (36 tasks, 195 requests
that follow a reply), 15 requests found the cached tokens differing from the new prompt inside the previous reply.
In 11 of them the difference lay more than the live window ring reaches (~255 tokens before the end at `-ub 128`):
the server restored the checkpoint at the previous prompt's end and processed the whole reply again.

| run | turn pairs | re-reads (checkpoint restores) | where the reply diverged | re-read tokens | re-read s | task time s (12 tasks) | share of task time | share of prompt time | worst first token s | same, if exact |
|---|---|---|---|---|---|---|---|---|---|---|
| d4 | 69 | 4 | reasoning | 16,954 | 94.4 | 1599 | 5.9 % | 31 % | 59.8 | 0.5 |
| q8-d4b | 65 | 5 | reasoning | 29,083 | 150.3 | 1818 | 8.3 % | 43 % | 80.9 | 0.9 |
| base | 61 | 2 | reasoning | 17,294 | 90.8 | 1360 | 6.7 % | 32 % | 82.5 | 3.5 |

- Every one of the 11 diverged in the middle of the reasoning text, at a plain word junction ("…`raw-tonic`| comparison",
  "…`0x1FFFFFF`|."), never at the template's markup. The other 4 differed 111–213 tokens before the end, 2 of them
  inside a tool call's argument text; the live ring served those (1.4–1.7 s each).
- The chat template re-renders every reply as the text Flint wrote: the 195 replies' rendered lengths add up to
  159,363 tokens against 159,598 generated. The 180 replies the cache matched to their end render to exactly their
  generated length less the stop token (which is not part of the history); the 15 with a difference render 0–22
  tokens shorter, as merged splits would. The tool calls come back byte-exact, `edit` arrays included.
- Per-turn rows: [turns/](turns/) (`div_in_reply`: the divergence's offset in the reply, `div_from_end`, `segment`,
  `reread_tokens`, `reread_ms`; the offset is known to ±0.05 % of the context, the log's three decimals).

**2. Why: Flint samples non-canonical token splits.** At temperature 1 the model sometimes writes a word in other ids
than its tokenizer would: `RAW` + `DESC` where the tokenizer writes `RA` + `WD` + `ESC`, `file` + `type` where it
writes `filetype`, `` ` `` + `tell` where it writes `` `t `` + `ell`. The client sends the reply back as text, the
template renders it again, and the tokenizer splits it its own way, so the token match with the cache ends at the
first such split. Data run 1's exact generated ids (Flint Q8_0, 1,247 turns, [retok.tsv](retok.tsv)), detokenized
and tokenized again:

| reply length (tokens) | turns | generated tokens | ids change in a text round trip | first change ≥ 256 tokens before the end | in the reasoning | in a tool call |
|---|---|---|---|---|---|---|
| 0–255 | 730 | 53,927 | 5 (0.7 %) | 0 | 0 | 5 |
| 256–1023 | 249 | 135,141 | 9 (3.6 %) | 7 | 5 | 3 |
| 1024–4095 | 199 | 409,707 | 31 (15.6 %) | 28 | 26 | 5 |
| ≥ 4096 | 69 | 527,055 | 31 (44.9 %) | 30 | 29 | 2 |
| all | 1,247 | 1,125,830 | 76 (6.1 %) | 65 | 60 | 15 |

So neither a template change nor a checkpoint before the tool call removes the re-reads: the text is right, the ids
are not, and the split can fall anywhere in a long reply.

**3. The fix (Q260): `LLAMA_SERVER_TEXT_REUSE=1`.** When a request's ids differ from the slot's cache, the server keeps
the cached ids past the first difference for as long as both spell the same bytes, cut where a token ends in both
sequences, so the prompt's text is unchanged and the model reads its own ids, as it did when it wrote them. Fork branch
`pascal-p106-v3-textreuse` (`60615dffd`, on `pascal-p106-v3`), off by default. CPU check on a tiny random Flint
(Flint's tokenizer and template, `-fa off`, the rig's `-b 512 -ub 128 --ctx-checkpoints 4`): in 6 of 6 trials the
700-token reply's ids differed from its text's tokenization; without the switch the next request restored the
18-token checkpoint and processed 744–761 tokens, with it the cache covered the prompt and the whole reply (720–721
tokens, 6–8 processed), and the answer equals a cold server's answer to the same ids. Again with 1,100-token replies,
past the 768-cell window ring: 4 of 4 ([check/](check/)).

**4. Per-request fixed cost (Q263).** Two of the pieces the plan listed are already the v2.0 defaults: one
end-of-prompt checkpoint holding the whole window ring (`--checkpoint-tail2` restores stock's two) and no scheduler
re-reserve when a request attaches the same sampler chain (`LLAMA_SAMPLER_RESERVE=1` restores stock). What a request
still pays: the end-of-prompt checkpoint (the pipeline drains and the ring, 52.5 MiB, is copied to the host; restoring
the same state takes 305 ms on the rig) and the separate micro-batch of the prompt's last 4 tokens that it needs. A
19-token turn costs 0.63–0.68 s of prompt time on the release line. `LLAMA_SERVER_TAIL_CKPT_MIN=<n>` takes that
checkpoint only when the prompt ends at least n tokens past the slot's newest one (user-message checkpoints are
unchanged); with text reuse a reply sent back unchanged needs none. CPU check with 4,096: the second request created
no checkpoint and every answer was the same ([check/textreuse_check_tail.tsv](check/textreuse_check_tail.tsv)).

**5. The decode-only card mask (Q261): `LLAMA_GPU_KQ_MASK_MAX=<m>`.** `LLAMA_GPU_KQ_MASK=<n>` turns the card-built
mask on for micro-batches of at least n tokens, so decode steps could not have it without prompt micro-batches (8k
prompts 3.6× slower on nine cards). The new variable limits it to micro-batches of at most m tokens:
`LLAMA_GPU_KQ_MASK=1 LLAMA_GPU_KQ_MASK_MAX=16` keeps decode steps and short verify batches on the card mask and
prompts on the host mask. After each host-mask prompt the cards' cell copies are stale and the first decode step
uploads all cells once. Fork branch `pascal-p106-v3-decodemask` (`55143f751`). On the CPU, `test-llama-archs` passes
141 of 141 with the host mask, the decode-only setting and the all-batches setting (the last compared 1,305 card masks
with the host's, all identical), but its micro-batches are 32–128 tokens, and llama-server on a CPU-only build keeps no
card copies. **On two GPUs** (two RTX 3080s, the rig tarball's compute 6.1 code compiled for them at load, the tiny
Flint layer-split over both cards, `-fa on`, 1,100-token replies past the 768-cell window ring, every card mask
compared with the host mask): 9,024 card masks identical to the host's, both cards and both caches (8,984 one-token
decode steps, 24 four-token prompt tails, 16 two-token steps; 24 full uploads after host-mask prompts), no micro-batch
over 16 tokens on the card mask, and the card-mask server samples the same 1,100-token replies as the host-mask server
from the same seeds in 6 of 6 trials ([check/textreuse_check_gpu_mask.tsv](check/textreuse_check_gpu_mask.tsv),
[check/textreuse_check_gpu_mask_v.tsv](check/textreuse_check_gpu_mask_v.tsv)).

**6. On the rig: not run yet.** The three switches are combined on `pascal-p106-v3-promptpath` (`8130a649d`), all off
by default, one tarball for both scripts:

| script | arms | measures | expected (est., from the numbers above) | rig time |
|---|---|---|---|---|
| [rig_agent.sh](run/rig_agent.sh) (live agent check) | `tr` (text reuse), `trtail` (+ tail checkpoints every 4,096 tokens) | restores, prompt s per task, first-token times, time per task | `tr`: 5.9–8.3 % less task time, worst first token 60–83 s → 0.5–3.5 s; `trtail`: ~0.3–0.4 s per request, 1.5–2 % of task time | ~70 min |
| [rig_replay_mask.sh](run/rig_replay_mask.sh) (paired replay, identical tokens) | `base`, `mask` | ms per generated token, fixed cost per request | +1–2 % per generated token (Q116: +1.6–2.3 % at short context; one cell upload per request, ≤ 0.1 s) | ~50 min |

The live check is needed for Q260 and Q263: a replay sends the recorded replies back, never the server's own, so
every replayed turn re-reads its reply whatever the server does.

## Files

| File | What it holds |
|---|---|
| [summary.md](summary.md) | Tables 1 and 2 as `summary.py` printed them |
| [turns/d4.tsv](turns/d4.tsv), [turns/q8-d4b.tsv](turns/q8-d4b.tsv), [turns/base.tsv](turns/base.tsv) | Per request that follows a reply: the render check against the server's prompt length, generated and rendered reply tokens, the shared prefix from the log, where it ends in the reply, the reply's parts, the next request's cache, restore and re-read cost |
| [retok.tsv](retok.tsv) | Data run 1, per turn: generated and re-tokenized lengths, the first differing id (−1: none), its distance from the end and the part of the reply |
| [check/](check/) | The CPU checks (text reuse with 700-token replies, with the tail switch, with 1,100-token replies) and the GPU check of the decode-only card mask (`_gpu_mask`: 4 trials; `_gpu_mask_v`: 2 more with the per-mask log lines counted) |
| [run/rereads.py](run/rereads.py) | Per-turn divergence from a run's `proxy.jsonl`, `proxy.jsonl.bodies.gz` and server log, rendered and tokenized by the server's own code |
| [run/retok.py](run/retok.py) | The text round trip of an agentic export's generated ids |
| [run/summary.py](run/summary.py) | Tables 1 and 2 |
| [run/build_cpu.sh](run/build_cpu.sh), [run/serve_tiny.sh](run/serve_tiny.sh) | The CPU llama-server (v2.0 or the fix branch) and the tokenizer-only server on the tiny Flint |
| [run/check_server.sh](run/check_server.sh), [run/textreuse_check.py](run/textreuse_check.py) | The server checks (CPU; GPU with `GPUS`) |
| [run/mask_check_cpu.sh](run/mask_check_cpu.sh) | `test-llama-archs` with the three mask settings |
| [run/rig_agent.sh](run/rig_agent.sh), [run/rig_replay_mask.sh](run/rig_replay_mask.sh) | The rig runs |

## Commands

```bash
# the v2.0 server (fork 37e717ff6, `git archive` into $WORK/src) on the tiny Flint, for /apply-template and /tokenize
WORK=$WORK bash run/build_cpu.sh
WORK=$WORK MODEL=$MODELS/tiny-flint.gguf bash run/serve_tiny.sh          # bench/agent_cache/make_tiny_flint.py
for r in d4 q8-d4b base; do   # each run's proxy.jsonl, proxy.jsonl.bodies.gz, rig/server-rel.log in $WORK/runs/$r
  python3 -I run/rereads.py --run $WORK/runs/$r --out $WORK/out/$r.tsv
done
python3 -I run/retok.py --turns $DATA/train/turns.jsonl --out $WORK/out/retok.tsv   # bench/agentic_data export
python3 -I run/summary.py --runs d4=out/d4.tsv,runs/d4/attempts.csv q8-d4b=out/q8-d4b.tsv,runs/q8-d4b/attempts.csv \
  base=out/base.tsv,runs/base/attempts.csv --retok out/retok.tsv > summary.md
# the fix branch on the CPU (pascal-p106-v3-promptpath archived into $WORK/src-pp)
WORK=$WORK SRC=$WORK/src-pp OUT=$WORK/out-cpu-pp bash run/build_cpu.sh
WORK=$WORK MODEL=$MODELS/tiny-flint.gguf bash run/check_server.sh
WORK=$WORK MODEL=$MODELS/tiny-flint.gguf ON_EXTRA=LLAMA_SERVER_TAIL_CKPT_MIN=4096 TAG=_tail bash run/check_server.sh
WORK=$WORK SRC=$WORK/src-pp OUT=$WORK/out-cpu-pp-t TESTS=1 bash run/build_cpu.sh && WORK=$WORK bash run/mask_check_cpu.sh
# the rig tarball, the GPU check on two cards with it, and the rig runs
scripts/build_llamacpp_docker.sh fork/pascal-p106-v3-promptpath
WORK=$WORK MODEL=$MODELS/tiny-flint.gguf GPUS=0,1 BIN=$WORK/<tarball>/bin TRIALS=2 N_PREDICT=1100 ON_ARGS="-lv 4" \
  ON_EXTRA="LLAMA_GPU_KQ_MASK=1 LLAMA_GPU_KQ_MASK_MAX=16 LLAMA_GPU_KQ_MASK_CHECK=1" TAG=_gpu_mask_v bash run/check_server.sh
ENV_FILE=... BIN=/opt/llama-builds/<tarball>/bin/llama-server CLIENT_ROOT=... bash run/rig_agent.sh
ENV_FILE=... BIN=... REPLAY_LOG=<the answer store control run's proxy.jsonl> CLIENT_ROOT=... bash run/rig_replay_mask.sh
```

All scripts run in Docker with the label `pascal-quants=1` at `--cpu-shares 256`, Python with the standard library only.

## Provenance and licence

- The agent check runs are the release-line runs of [2026-10-26 release setup](../2026-10-26_release_setup/README.md)
  (`d4`, `q8-d4b`) and [2026-10-27 MTP head](../2026-10-27_mtp_head/README.md) (`base`): Exercism tasks (MIT) and
  Flint's replies. Their request and reply bodies stay private; the tables here hold counts and positions only.
- Data run 1 ([bench/agentic_data](../../../bench/agentic_data/README.md)): Flint Q8_0's sessions on licence-scanned
  code; `retok.tsv` holds session ids, turn numbers and counts, no text.
- The tiny Flint has random weights and Flint's own tokenizer and template (`bench/agent_cache/make_tiny_flint.py`);
  all 231 rendered prompts have exactly the length the rig's server reported, so it renders and tokenizes as Flint
  does.
- Engine: the fork's `pascal-p106-v3` (= v2.0, `37e717ff6`) and the branches above. Tables and scripts: this
  repository's licence.
