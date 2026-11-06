# Ornith's own MTP draft on a second rig: the control, `mtpka` and `mtpadapt` (2026-11-06)

The agent check (`bench/agent_check/release_check.sh`: a coding agent fixing 12 small bugs with real tool runs, one
attempt each, 12-minute cap, 35-minute budget per arm) with Ornith-1.5-122B Pascal N2P GGUF (`valid1`) on nine P106-090
cards, on the v3 release build `-ladder5` `1551f5fe1`. Every arm runs a 49,152-token context, the harness's
`-np 1 -bs --ctx-checkpoints 4 --cache-ram 0` and Ornith's makers' sampling (T 0.6, top-p 0.95, top-k 20, min-p 0).
Part 1 runs the control and the first MTP arm; `mtpadapt` follows. All three ran on the second rig the same
afternoon:

| Arm | Server line | Environment |
|---|---|---|
| `ka` (control) | the preset: `-b 2048 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off` | `GGML_CUDA_KEEPALIVE=decode` |
| `mtpka` | `-b 2048 -ub 256 -ts 6,6,6,6,6,5,5,5,5 -fit off --spec-type draft-mtp --spec-draft-n-max 2`: the model's own MTP layer drafts two tokens a step | `GGML_CUDA_KEEPALIVE=decode` |
| `mtpadapt` | the same as `mtpka` | `LLAMA_SPEC_ADAPT=1` with `LLAMA_SPEC_ADAPT_COST` the T(k) table measured on this line's split (below): the draft length chosen per step; no keep-alive |

It continues [2026-11-05_release_regression](../2026-11-05_release_regression/README.md), where the preset ran on the
first rig. The MTP layer and the multi-token kernels it relies on: [2026-10-28_multitoken_kernels](../2026-10-28_multitoken_kernels/README.md)
(registry row Q211). Result page: [release/ornith-v3/README.md](../../../release/ornith-v3/README.md).

## Results

| Arm | Passed | Gen tok/s | Prompt tok/s | TTFT median, p90 s | Minutes, all tasks | Min a task (median) | Draft acceptance, tokens a step | Equal work, min a task |
|---|---|---|---|---|---:|---:|---|---|
| `ka` | 12/12 | 16.42 | 60.1 | 9.34, 19.86 | 30.2 | 2.11 | — | 2.52 |
| `mtpka` | 8/9 (3 not started: budget) | 13.01 | 39.4 | 12.61, 38.44 | 39.5 | 2.91 | 0.571, 2.14 | 3.42 (+35.7 %) |
| `mtpadapt` | 11/11 (1 not started: budget) | 12.78 | 40.0 | 11.43, 31.43 | 38.0 | 3.52 | 0.744, 1.99 | 3.44 (+36.5 %) |

The equal-work column times one fixed amount of work for every arm: `ka`'s prompt tokens at the arm's prompt speed,
plus `ka`'s generated tokens at the arm's generation speed, plus `ka`'s time outside the server, divided over its 12
tasks ([agent_tables.py](../2026-11-05_keepalive/run/agent_tables.py)). On the 9 tasks both part 1 arms ran it is 3.01
against 2.23 minutes a task (+35.0 %, `agent_part1_9tasks.tsv`); on the same 9 tasks `mtpadapt` takes 3.04 (+36.3 %,
`agent_mtpadapt_9tasks.tsv`), and on the 11 tasks it ran 3.38 against 2.47 (+36.8 %, `agent_mtpadapt_11tasks.tsv`).
`mtpadapt`'s tokens a step are the server's own count, 1.99 (17,720 decode tokens in 8,905 steps, `step_costs.tsv`;
the rule's own log gives 1.987); the tables'
`tokens_per_step` column (1 + n-max × acceptance, 2.49 for `mtpadapt`) assumes every step verifies two drafted tokens,
which holds for `mtpka` only.

**The MTP draft (`mtpka`): a loss.** At equal work it takes 3.42 against 2.52 minutes a task (+35.7 %).

- **Generation:** 13.01 against 16.42 tok/s. Over the session 8,821 of 15,452 drafted tokens were accepted (0.571;
  per request in `agent/ornith-mtpka/draft_acceptance.txt`), so a step yields 2.14 tokens. On these cards a step that
  verifies three tokens costs more than the 2.14 tokens it yields.
- **Prompt:** 39.4 against 60.1 tok/s, from the half-size micro-batch the MTP line needs to fit (`-ub 256`) and the
  draft context's own pass over every prompt.
- **Time to first token:** 12.61 against 9.34 s (median) and 38.44 against 19.86 s (p90).
- **Tasks:** the arm ran out of its 35-minute budget after 9 tasks. Of those, 8 passed and one hit the 12-minute cap
  (`bugfix-python-scale-generator`; it passed in `ka`).

One run each: the spread of repeated loads of `ka` on this rig is measured in part 2. The first rig's `ka` on the build
before (`1f7215948`) gave 16.3 tok/s, 12/12, 8.39 s first token (median), so a 35 % gap is far beyond it.

**The draft length chosen per step (`mtpadapt`): a loss too.** At equal work it takes 3.44 against 2.52 minutes a
task (+36.5 %), the same as `mtpka`'s 3.42.

- **The rule** (`LLAMA_SPEC_ADAPT`, registry Q687): after each draft it keeps the longest prefix whose path
  probabilities, from the drafter's own probabilities, pay for the wider verify on the T(k) table. With n-max 2 the
  verify widths allowed are 1 and 3, so a step verifies two drafted tokens or none.
- **What it did:** of 8,933 drafted steps it kept both drafted tokens on 5,925 and cut 3,008 (33.7 %) to one-token
  steps (`agent/ornith-mtpadapt/adapt_summary.csv`). Of the tokens it verified, 0.744 were accepted (`mtpka`: 0.571),
  but a step yields 1.99 tokens against `mtpka`'s 2.14. `adapt_log.py --check` recomputes every step's choice from its
  probabilities and the table: 0 problems.
- **Generation:** 12.78 against 16.42 tok/s (`mtpka` 13.01, with the keep-alive). From the server's own per-request
  times (`step_costs.tsv`), `ka` takes 61.1 ms a token, `mtpka` 165.1 ms a step (77.0 ms a token) and `mtpadapt` 156.2
  ms a step (78.5 ms a token).
- **Why cutting does not pay here:** the rule decides after the draft pass, so a cut step still pays for the MTP
  layer's two draft tokens. If `mtpadapt`'s three-token steps cost what `mtpka`'s steps did, a cut step cost about
  138 ms (est.), more than two of `ka`'s decode steps. The rule's inputs were further off: its table puts a one-token
  pass at 104.6 ms (llama-bench's one-token prompt pass) against the server's 61.1 ms decode step, and its draft cost is
  the engine's default 20 ms. It modelled a 1.073× speed-up over one-token decoding.
- **Prompt and first token:** 40.0 against 60.1 tok/s; 11.43 against 9.34 s (median) and 31.43 against 19.86 s (p90),
  both a little better than `mtpka`'s.
- **Tasks:** 11 of 11 passed in the 35-minute budget, and one task (`bugfix-javascript-variable-length-quantity`)
  never started. `bugfix-python-scale-generator`, which hit the 12-minute cap in `mtpka`, passed in 548 s.
- **Fit:** the cards peaked at 5,407, 5,465, 5,389, 5,465, 6,037, 5,237, 5,417, 5,343 and 5,031 MiB (card 4 at 6,037
  of 6,068). The host's lowest MemAvailable was 614 MiB (`ka` 736, `mtpka` 739), and swap went from 30 to 334 MiB.

Both MTP arms load without pipeline parallelism: on this split the first compute-buffer reserve fails on card 4
(170.80 MiB), and the scheduler retries without it (`sched_reserve: compute buffer allocation failed, retrying without
pipeline parallelism`). `ka`'s log has no such line. Its effect is on the prompt pass, which is part of the 40 against
60 tok/s gap; generation runs one token sequence, so the 12.78 and 13.01 against 16.42 tok/s do not come from it.

### T(k) on the MTP line's split

The table `mtpadapt` ran on, from llama-bench's k-token prompt passes on the MTP line's `-b 2048 -ub 256
-ts 6/6/6/6/6/5/5/5/5`, five repetitions each, T(k) = 1000 k / tokens per second (`tk_mtp_split.tsv`):

| k | T(k) ms | Mean ms (sd) | The five passes, ms |
|---:|---:|---|---|
| 1 | 104.6 | 105.3 (9.7) | 122.6, 99.5, 100.5, 101.8, 101.9 |
| 2 | 183.9 | 184.0 (5.7) | 194.2, 181.8, 181.8, 181.9, 180.5 |
| 3 | 208.7 | 212.3 (32.0) | 257.6, 189.6, 189.9, 234.8, 189.6 |
| 4 | 293.3 | 294.2 (18.1) | 319.0, 279.9, 294.6, 274.0, 303.4 |

## The fit: the MTP layer goes to the last card

With the MTP draft on, the last card holds the output layer, the MTP layer and the draft context. On the preset's split
(`-ts 6,6,6,6,5,5,5,5,6`) it started at 5,457 MiB against 4,787–5,089 MiB on the others and ran out of memory on the
first request (`ggml_cuda_pool_vmm::alloc` in a quantized matrix product). Moving one share from the last card to card
4, the lightest, fits: `-ts 6,6,6,6,6,5,5,5,5`. Over the session the cards peaked at 5,405, 5,463, 5,387, 5,463, 6,035,
5,233, 5,415, 5,341 and 5,029 MiB, so card 4 is then the fullest, at 6,035 of the 6,068 MiB a card has. The `ka` arm's cards peaked at
5,665, 5,697, 5,611, 5,697, 5,305, 5,413, 5,575, 5,663 and 5,069 MiB.

## The model file on the rig

`Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`, 48,196,722,592 bytes, md5 `7985df46ec7bb3867e482eaf91aaef39`, taken
on the rig over the whole file as it was written (the copy resumed a first copy that had stopped at 14,777,761,792
bytes: the part already on the stick was read back into the checksum first). It came from the quant machine over the
LAN with `nc`; the rest took 1,742 s. It sits on the second rig's 128 GB USB 3 stick, which had 10,289,537,024 bytes
free before Unsloth's Flint Q8_0 copy (42,301,076,928 bytes, its original kept elsewhere) was deleted, 52,590,632,960
after, and 4.1 GiB once `valid1` was in (`df -B1`, `df -h`).

## Files

| File | What |
|---|---|
| `agent_part1.tsv` | both arms side by side: passed, turns, prompt and generation tok/s, first token per turn (median, p90, first turn), draft acceptance and tokens a step, minutes, equal work against `ka` |
| `agent_part1_9tasks.tsv` | the same on the 9 tasks both arms ran |
| `agent_mtpadapt.tsv` | all three arms side by side, the same columns (`tokens_per_step` holds for `mtpka` only: see above) |
| `agent_mtpadapt_9tasks.tsv`, `agent_mtpadapt_11tasks.tsv` | the same on the 9 tasks all three arms ran, and `ka` against `mtpadapt` on the 11 tasks `mtpadapt` ran |
| `step_costs.tsv` | per arm from the server's per-request lines: requests, decode tokens, steps, tokens a step, ms a token, ms a step, and `mtpadapt`'s steps by verify width ([run/step_costs.py](run/step_costs.py)) |
| `tk_mtp_split.tsv` | the T(k) table `mtpadapt` ran on, with each pass |
| `agent/ornith-<arm>/` | each arm's report: `compare.md` (per task, first token per turn), `table.md`, `gonogo.md`, `attempts.csv`, `turns.csv`; `validation.txt` (the task set checked in the sandbox image) with `ka`; `draft_acceptance.txt` with `mtpka` and `mtpadapt`: the server's draft acceptance per request; `adapt_summary.csv` with `mtpadapt`: the rule's log summarized ([adapt_log.py](../2026-11-04_adaptive_draft/run/adapt_log.py)) |

## How it was made

Run from the repository root, nothing else on the rig, the client (Docker, the agent and the tasks' tests) on another
machine. Each arm is one call of [agent_ornith.sh](../2026-10-27_ornith/run/agent_ornith.sh) with its own client
folder. `PREFIX` names the client's containers, so arms of another rig that share the client keep theirs.

```bash
# the machine values as in bench/agent_check/release.env.example: RIG, CLIENT, RIG_PORT, RELAY_PORT, CLIENT_PORT,
# PROXY_PORT; the model and build on the rig
export PREFIX=acr2o
B=<llama.cpp-20261105-1551f5fe1-pascal-p106-v3-b12076-ladder5>/bin/llama-server
M=<rig path>/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf
# the model onto the rig, its md5 taken there (resuming: the part already there is read back into the checksum first)
ssh $RIG "nc -l <rig address> <port> | { cat $M.part; tee -a $M.part; } | md5sum > copy.md5" &
tail -c +$((HAVE + 1)) Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf | nc -N <rig address> <port>
# ka, the control
REL_ENV=GGML_CUDA_KEEPALIVE=decode CLIENT_WORK=<client folder>/ka AD=unused STOCK=unused V2=unused LOWBIT=$B \
  VALID1=$M VALID1_LINE="-b 2048 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off" OUT=agent_ka CANDS=valid1 \
  bash docs/data/2026-10-27_ornith/run/agent_ornith.sh
# mtpka
REL_ENV=GGML_CUDA_KEEPALIVE=decode CLIENT_WORK=<client folder>/mtpka AD=unused STOCK=unused V2=unused LOWBIT=$B \
  VALID1=$M VALID1_LINE=unused \
  VALID1_MTP_LINE="-b 2048 -ub 256 -ts 6,6,6,6,6,5,5,5,5 -fit off --spec-type draft-mtp --spec-draft-n-max 2" \
  OUT=agent_mtpka CANDS=valid1-mtp bash docs/data/2026-10-27_ornith/run/agent_ornith.sh
# the side by side, from the two client folders
python3 docs/data/2026-11-06_flint_side_rig2/run/merge_arms.py m ka=<client folder>/ka/out-ornith-valid1 \
  mtpka=<client folder>/mtpka/out-ornith-valid1-mtp
NMAX=2 python3 docs/data/2026-11-05_keepalive/run/agent_tables.py m ka ka mtpka   # m/agent.tsv: agent_part1.tsv
NMAX=2 OUT_TAG=_9tasks SKIP_TASKS=bugfix-javascript-complex-numbers,bugfix-python-pig-latin,bugfix-javascript-variable-length-quantity \
  python3 docs/data/2026-11-05_keepalive/run/agent_tables.py m ka ka mtpka      # agent_part1_9tasks.tsv
grep "draft acceptance" m/rig/server-mtpka.log   # draft_acceptance.txt, without the timestamps
# mtpadapt: T(k) on the MTP line's split first (the table stays on the rig as <rig folder>/tk/t_k.txt), then the arm
RIG=$RIG MODEL=$M BIN=<llama.cpp-20261105-1551f5fe1-pascal-p106-v3-b12076-ladder5>/bin/llama-bench \
  ARGS="-b 2048 -ub 256 -ts 6/6/6/6/6/5/5/5/5" DIR=<rig folder>/tk OUT=tk \
  bash docs/data/2026-11-05_release_regression/run/tk_cost.sh
REL_ENV="LLAMA_SPEC_ADAPT=1 LLAMA_SPEC_ADAPT_LOG=adapt-mtpadapt.jsonl LLAMA_SPEC_ADAPT_COST=<rig folder>/tk/t_k.txt" \
  CLIENT_WORK=<client folder>/mtpadapt AD=unused STOCK=unused V2=unused LOWBIT=$B VALID1=$M VALID1_LINE=unused \
  VALID1_MTP_LINE="-b 2048 -ub 256 -ts 6,6,6,6,6,5,5,5,5 -fit off --spec-type draft-mtp --spec-draft-n-max 2" \
  OUT=agent_mtpadapt CANDS=valid1-mtp bash docs/data/2026-10-27_ornith/run/agent_ornith.sh
# all three side by side
python3 docs/data/2026-11-06_flint_side_rig2/run/merge_arms.py m ka=<client folder>/ka/out-ornith-valid1 \
  mtpka=<client folder>/mtpka/out-ornith-valid1-mtp mtpadapt=<client folder>/mtpadapt/out-ornith-valid1-mtp
NMAX=2 python3 docs/data/2026-11-05_keepalive/run/agent_tables.py m ka ka mtpka mtpadapt   # agent_mtpadapt.tsv
NMAX=2 OUT_TAG=_9tasks SKIP_TASKS=bugfix-javascript-complex-numbers,bugfix-python-pig-latin,bugfix-javascript-variable-length-quantity \
  python3 docs/data/2026-11-05_keepalive/run/agent_tables.py m ka ka mtpka mtpadapt         # agent_mtpadapt_9tasks.tsv
NMAX=2 OUT_TAG=_11tasks SKIP_TASKS=bugfix-javascript-variable-length-quantity \
  python3 docs/data/2026-11-05_keepalive/run/agent_tables.py m ka ka mtpadapt               # agent_mtpadapt_11tasks.tsv
python3 docs/data/2026-11-06_ornith_mtp_rig2/run/step_costs.py ka=m/rig/server-ka.log mtpka=m/rig/server-mtpka.log \
  mtpadapt=m/rig/server-mtpadapt.log                                                         # step_costs.tsv
python3 docs/data/2026-11-04_adaptive_draft/run/adapt_log.py --check --csv adapt_summary.csv \
  <client folder>/mtpadapt/out-ornith-valid1-mtp/rig/adapt-mtpadapt.jsonl
```

**Provenance and licence:** the tables summarize runs of Ornith-1.5-122B Pascal N2P GGUF (`valid1`, made from
Ornith-1.5-122B under its licence: [2026-10-28_ornith_valid1](../2026-10-28_ornith_valid1/README.md)) on Miner Llama's
`pascal-p106-v3-b12076-ladder5` build at `1551f5fe1`. The texts the agent check runs on are its Exercism-based bug-fix
tasks, built by [build_bugfix.py](../../../bench/agent_check/build_bugfix.py) under their MIT licence.
