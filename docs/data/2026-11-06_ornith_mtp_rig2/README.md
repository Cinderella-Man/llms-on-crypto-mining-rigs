# Ornith's own MTP draft on a second rig, part 1: the control and `mtpka` (2026-11-06)

The agent check (`bench/agent_check/release_check.sh`: a coding agent fixing 12 small bugs with real tool runs, one
attempt each, 12-minute cap, 35-minute budget per arm) with Ornith-1.5-122B Pascal N2P GGUF (`valid1`) on nine P106-090
cards, on the v3 release build `-ladder5` `1551f5fe1`. Every arm runs a 49,152-token context, the harness's
`-np 1 -bs --ctx-checkpoints 4 --cache-ram 0` and Ornith's makers' sampling (T 0.6, top-p 0.95, top-k 20, min-p 0).
Part 1 runs the control and the first MTP arm, both on the second rig the same afternoon:

| Arm | Server line | Environment |
|---|---|---|
| `ka` (control) | the preset: `-b 2048 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off` | `GGML_CUDA_KEEPALIVE=decode` |
| `mtpka` | `-b 2048 -ub 256 -ts 6,6,6,6,6,5,5,5,5 -fit off --spec-type draft-mtp --spec-draft-n-max 2`: the model's own MTP layer drafts two tokens a step | `GGML_CUDA_KEEPALIVE=decode` |

It continues [2026-11-05_release_regression](../2026-11-05_release_regression/README.md), where the preset ran on the
first rig. The MTP layer and the multi-token kernels it relies on: [2026-10-28_multitoken_kernels](../2026-10-28_multitoken_kernels/README.md)
(registry row Q211). Result page: [release/ornith-v3/README.md](../../../release/ornith-v3/README.md).

## Results

| Arm | Passed | Gen tok/s | Prompt tok/s | TTFT median, p90 s | Minutes, all tasks | Min a task (median) | Draft acceptance, tokens a step | Equal work, min a task |
|---|---|---|---|---|---:|---:|---|---|
| `ka` | 12/12 | 16.42 | 60.1 | 9.34, 19.86 | 30.2 | 2.11 | — | 2.52 |
| `mtpka` | 8/9 (3 not started: budget) | 13.01 | 39.4 | 12.61, 38.44 | 39.5 | 2.91 | 0.571, 2.14 | 3.42 (+35.7 %) |

The equal-work column times one fixed amount of work for both arms: `ka`'s prompt tokens at the arm's prompt speed,
plus `ka`'s generated tokens at the arm's generation speed, plus `ka`'s time outside the server, divided over its 12
tasks ([agent_tables.py](../2026-11-05_keepalive/run/agent_tables.py)). On the 9 tasks both arms ran it is 3.01 against
2.23 minutes a task (+35.0 %, `agent_part1_9tasks.tsv`).

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
| `agent/ornith-<arm>/` | each arm's report: `compare.md` (per task, first token per turn), `table.md`, `gonogo.md`, `attempts.csv`, `turns.csv`; `validation.txt` (the task set checked in the sandbox image) with `ka`; `draft_acceptance.txt` with `mtpka`: the server's draft acceptance per request |

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
```

**Provenance and licence:** the tables summarize runs of Ornith-1.5-122B Pascal N2P GGUF (`valid1`, made from
Ornith-1.5-122B under its licence: [2026-10-28_ornith_valid1](../2026-10-28_ornith_valid1/README.md)) on Miner Llama's
`pascal-p106-v3-b12076-ladder5` build at `1551f5fe1`. The texts the agent check runs on are its Exercism-based bug-fix
tasks, built by [build_bugfix.py](../../../bench/agent_check/build_bugfix.py) under their MIT licence.
