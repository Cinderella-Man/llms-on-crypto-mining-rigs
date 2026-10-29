# A cheaper guess of Flint's MTP head (2026-10-27)

The data behind [A cheaper guess](../../drafter.md#a-cheaper-guess-2026-10-27) in docs/drafter.md (section 6, and
section 8, item 0): where one guess of the
staircase head (Flint's MTP head of agentic run 1, the nine-card head) goes on a P106, what a guess may cost on the
agent check before the head stops paying, and how often the head is right at the positions the server's other
guessers leave it; then what the options of Miner Llama's branch `pascal-p106-v3-cheapguess` cost (4.77 ms → 2.93 ms
per guess) and that they compute what the reference computes; and the agent check as a paired replay (2026-10-28): with
the overnight-trained head the release line is **3.2 % slower** per generated token, though the head is right 24.8 % of
the time as predicted. Registry rows Q190–Q194 in [quants/experiments.md](../../../quants/experiments.md) (profile,
break-even and guess order, options, the agent-check replay, the next cuts).

| File | What |
|---|---|
| `guess_cost.tsv` | every timing of one guess on one P106 (`llama-pipe-drafter-check --bench`): build, option set and its environment, ring cells, tokens in flight, commits, guesses timed, median / p90 / mean microseconds from the launch to the guess on the host, and the launch call alone |
| `profile-v20-ctx2048.txt`, `profile-v20-ctx8192.txt` | the per-kernel GPU trace of Miner Llama v2.0's guess (`GGML_CUDA_PROFILE`, last 80 guesses) at 4,352 and 8,448 cells, summarized by [scripts/cuda_profile_summary.py](../../../scripts/cuda_profile_summary.py) |
| `host-v20-ctx2048.txt`, `host-v20-ctx8192.txt` | the per-evaluation host and GPU trace (`GGML_CUDA_PROFILE_HOST`, last 200 guesses): GPU time per guess and the host's time inside `graph_compute` |
| `profile-q8-ctx2048.txt`, `profile-q8mmq-ctx2048.txt`, `profile-q8split-ctx2048.txt` | the per-kernel trace of the cheap-guess build at 4,352 cells with the Q8_0 output layer, then also MMQ for the 8-row products, or the products in groups of 4 rows |
| `served_guesses.tsv` | the agent check of the first heads (2026-10-27, [section 6](../../drafter.md#m4-heads-on-the-agent-check-2026-10-27)), llama-server's guess counts summed per arm: checked guesses and right ones, of them the head's and the lookups' (copy lookup and continuations) |
| `break_even.tsv` | [cost_sim.py](../../../bench/staircase_drafter/cost_sim.py) on the agent check's 204 requests: generation tok/s and the guesses per source for every guess order, head time per guess and acceptance (card 1, then a spare card) |
| `source_split.md` | [source_split.py](../../../bench/staircase_drafter/source_split.py): the head's offline acceptance on the 40 held-out sessions, split by what the server's lookups would have guessed, and a gate on the head's confidence |
| `shortlist_coverage.txt` | [shortlist_coverage.py](../../../bench/staircase_drafter/shortlist_coverage.py): the share of the held-out sessions' generated tokens the first K draft ids cover |
| `check.txt` | the C++ head against `drafter.Drafter.forward` per option (random-weight head of the real shape, the release split's windows), on the CPU and on a P106 |
| `replay.md` | the paired replay (`replay.py summary`): per label the cost per prompt and per generated token, and the ratios to the release line with 95 % intervals over the six traces; `live:rel` is the recorded live run the traces come from |
| `replay_served.tsv` | llama-server's guess counts per replay arm (as `served_guesses.tsv`) |
| `replay_pair_check.txt` | `run/pair_check.py`: per trace and request the reply length, prompt tokens, cache hits, finish reason and tool calls agree across the two arms (0 of 36 differ) |
| `replay_overhead.txt` | `cost_sim.py` with a longer guess and an extra cost per token: what it takes to reproduce the replay's −3.2 % |
| `profile_replay.md` | Q194, where the served cost goes: the paired replay of five traces with three arms, the release line (`base`), the head loaded and following the sequence without guessing (`taps`), the head (`stair`), all with the host trace on |
| `profile_gpu.md` | `run/prof_compare.py` over the three arms' `GGML_CUDA_PROFILE_HOST` traces: GPU ms per generated token per card and graph evaluations |
| `q310_replay.md`, `q310_served.tsv`, `q310_preflight.txt` | Q310, the same paired replay on three RTX 3090s (compute 8.6, build `366478ef6` for sm_86, `-ts 17,16,16 -c 262144 --pipe-depth 3`): the release line, + the 3–12-card head, + the nine-card head (1,000 minutes each); the pair checks (0 of 36 differ); the guess counts; the logged quiet check before the start |
| `profile_served.tsv`, `profile_pair_check.txt` | the three arms' guess counts (`taps` guesses exactly as `base`), and the same-tokens check (0 of 29 requests differ) |
| `q384_agentcheck.md`, `q384/` | Q384, the heads offline on the agent check's own six traces (36 requests dumped like the training data): served against offline at the head's positions, per segment, Flint's top probability, context length and trace; per head and α (1, and 3 as served: `-a3`) the `score.py`, `source_split.py` and `breakdown.py` outputs |
| `q385_heldout.md`, `q385/` | Q385, the held-out sessions re-scored with Flint's top probability and the overlap: from held-out to served at the head's positions, and where the head is right (segment, source, Flint's top probability, context, request number); per head and α the `score.py` and `breakdown.py` outputs |
| `run/` | the scripts: `build.sh` (the rig build), `profile.sh`, `variants.sh` (rig), `sim.sh` (the offline steps), `agent.sh` (live agent-check arms), `replay.sh` (the paired replay of recorded agent-check traces, release line against the head on the same build), `pair_check.py`, `prof_compare.py` (GPU time per card from host traces), `q310.sh` (the same replay on three RTX 3090s, with a quiet check), `q384.sh` (the agent check's own traces dumped and scored), `q385.sh` (the held-out sessions at the nine-card windows with Flint's top probability and the overlap, split by `breakdown.py`), `q371.sh` (Q371 + Q372: the training arm against a control from the same head and steps, then their scores), `curate.py` (these tables from the raw outputs) |

## How it was made

- **Head:** the nine-card head of agentic run 1 after 240 minutes (`flint-mtp-nine.gguf`, 2 layers, 29,288 draft ids,
  F16 output layer, Q8_0 matrices; [section 6](../../drafter.md#m4-heads-on-the-agent-check-2026-10-27)).
- **Rig:** one P106-090 (the rig's first card, nothing else running). The window is the release split's (`-ts
  6,6,5,5,6,5,5,6,6` assigns 6/6/5/5/6/5/5/6/5 layers: L6, L12, L22, L33, L37, L44), 6 tokens in flight and 1 commit
  in an 8-row graph. `--ctx 2048` makes the bench's ring 4,352 cells, the size of llama-server's default
  `--pipe-draft-ctx 4096`; `--ctx 8192` makes 8,448.
- **Builds:** Miner Llama v2.0 (`37e717ff6`, the release) for the profile; branch `pascal-p106-v3-cheapguess` for the
  options (commits `5462d458a`, `72552ee12`), built by `run/build.sh` (CUDA 12.9, compute capability 6.1).
- **Agent check requests:** the three arms of the first heads' run (release line, nine-card head, 3–12-card head), 204
  requests: prompts and replies from the agent check's proxy log, rendered with Flint's own chat template
  (`agent_tokens.py`); the replies hold 129,890 tokens: in 203 requests the server's own count minus the stop token
  (113,440 counted), and one request of 16,662 tokens that the proxy log holds no count for.
- **Held-out sessions:** the 40 sessions of held-out modules of agentic run 1 and the head's per-position scores from
  `score.py` (inverse-CDF coupling, as served).
- **Paired replay (2026-10-28):** six recorded traces of the release line's agent check (the answer store's control
  arm: space-age, phone-number and complex-numbers in JavaScript, affine-cipher, list-ops and scale-generator in
  Python; 36 requests, 20,544 generated tokens, scale-generator ~57 % of them) replayed with the recorded prompts, a
  fixed seed per request and the recorded reply lengths to two llama-servers on the cheap-guess build, Flint Q8_0 on
  nine cards at 262,144 tokens, the release line `-b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4`, both with
  `GGML_CUDA_MMVQ_MAX_COLS=4`: `base` (the release line's guesser `cont`) and `stair` (+ `--pipe-guess staircase
  --pipe-draft` the nine-card head after 1,000 minutes of training, held-out 0.366 / fresh 0.323 at α 1;
  `LLAMA_DRAFT_HEAD=q8_0`). The first `stair` start ran out of host memory while converting the output layer (the
  conversion then held ~424 MB at once); it ran again on `1d07352bf`, which converts in 8 MB pieces and changes nothing
  else, so `base` (on `72552ee12`) and `stair` generate the same tokens (`replay_pair_check.txt`). The cache carries
  over between traces in the same order in both arms.
- **Where the served cost goes (Q194, 2026-10-28):** `run/replay.sh` with `ARMS_LIST="base taps stair" PROFILE=1`, five of
  the six traces (scale-generator left out: 29 requests, 7,383 generated tokens), build `366478ef6` (adds
  `LLAMA_PIPE_DRAFT_TAPS_ONLY=1`: the server loads the head and follows the sequence, the taps of every token and the
  prompt's final states, but never guesses and keeps the release line's guesses). The host trace costs a lot on the rig's
  2-core host: the release line takes 49.4 ms per generated token with it against 22.85 without, so the arms' ratios
  are larger than the served ones; the GPU times per card (`profile_gpu.md`) are the cleaner split.
- **The agent check's own text (Q384, 2026-10-29):** `run/q384.sh` on the six traces of the paired replay (the
  release line's recorded bodies): `agent_tokens.py --traces ... --id-base 900000` (rows `agentcheck_900001` …
  `agentcheck_900036`), `llama-staircase-dump` on two RTX 3090s with Flint Q8_0 (the 26-layer grid, `DUMP_ALLPOS=4,6,9`,
  `-c 66048`), `score.py --only-windows g9,g9miss` per head on one GPU, at α 1 and α 3 (`ALPHA=3`), then
  `source_split.py` and `breakdown.py`.
- **Held-out breakdowns (Q385, 2026-10-29):** `run/q385.sh` on the 40 held-out sessions (`--spans`, the dump of the
  training run), heads nine-card 1,000 and 1,480 minutes, `ALPHA=3` then `ALPHA=1`, one RTX 3090.
- Offline steps on a separate CPU machine (Docker, low CPU priority); every timing on the rig.

Commands (from the repository root of a checkout that reaches the rig and the build machine; the values below are
examples of what each variable names):

```bash
export RIG=rig BUILD_HOST=<build machine> BUILD_WORK=<folder there> SRC=<fork checkout> RIG_DIR=<rig folder>
SRC=$SRC bash docs/data/2026-10-27_cheap_guess/run/build.sh                                   # prints NAME and md5
BIN=<rig bin folder of v2.0> HEAD=<rig path of flint-mtp-nine.gguf> OUT=out/v20 \
    bash docs/data/2026-10-27_cheap_guess/run/profile.sh
GGUF_PY=... VOCAB=... SESSIONS=... IDS=... BODIES="..." FLINT_HF=... TOKENIZER_IMAGE=... ROWS=... SCORES=... \
    bash docs/data/2026-10-27_cheap_guess/run/sim.sh case coverage tokens cost split
NAME=<tarball name> HEAD=<rig path of flint-mtp-nine.gguf> OUT=out/new bash docs/data/2026-10-27_cheap_guess/run/variants.sh
ENV_FILE=<agent-check machine values> REPLAY_LOG=<recorded proxy.jsonl on the client> CLIENT_ROOT=<client folder> \
    NAME=<tarball name> HEAD=<rig path of flint-mtp-nine-1000min.gguf> OUT=out/replay \
    bash docs/data/2026-10-27_cheap_guess/run/replay.sh                                       # ARMS_LIST="stair base"
python docs/data/2026-10-27_cheap_guess/run/pair_check.py <client folder>/out-cheap-guess-replay/replay.jsonl
python docs/data/2026-10-27_cheap_guess/run/curate.py docs/data/2026-10-27_cheap_guess \
    --bench v2.0=out/v20/bench.log cheapguess=out/new/bench.log --served release=<server log> nine=... gen=... \
    --sim cost-agent.json cost-agent-spare.json
python docs/data/2026-10-27_cheap_guess/run/curate.py out/replay-tables \
    --served release-replay=<replay server log>@llama-server-base head-1000min-replay=<same log>@<build>/bin/llama-server-stair
```

The replay ran as `ARMS_LIST="stair base"` (the first `stair` start failed to load), then `ARMS_LIST=stair` on the fixed
build with the same client folder, so the report pairs the two labels:

```bash
ARMS_LIST=stair NAME=llama.cpp-20261028-1d07352bf-pascal-p106-v3-cheapguess ... bash docs/data/2026-10-27_cheap_guess/run/replay.sh
```

Q194 (the three arms with the host trace, five traces), then the per-card GPU times:

```bash
ARMS_LIST="base taps stair" PROFILE=1 REPLAY_DIR=out-cheap-guess-profile REPLAY_TRACES=<the six without scale-generator> \
    NAME=llama.cpp-20261028-366478ef6-pascal-p106-v3-cheapguess ... bash docs/data/2026-10-27_cheap_guess/run/replay.sh
python docs/data/2026-10-27_cheap_guess/run/prof_compare.py --tokens 7383 base=out/prof-base.csv taps=out/prof-taps.csv stair=out/prof-stair.csv
```

Q384 (one step at a time; `ALPHA=3` scores and splits again as served):

```bash
HOST=<GPU machine> WORK=<folder there> BODIES=<the replayed arm's proxy.jsonl.bodies.gz> FLINT_HF=<Flint's tokenizer folder> \
    MODEL=<Flint Q8_0> DUMP_BIN=<llama-staircase-dump> HEADS="<head folders>" VOCAB=<run>/vocab.npy FLINT_DIR=<run>/flint \
    TOKENIZER_IMAGE=<image with transformers> OUT=out/q384 STEPS=rows bash docs/data/2026-10-27_cheap_guess/run/q384.sh
...  STEPS=dump ...; STEPS=score ...; STEPS=split ...; ALPHA=3 STEPS="score split" ...
```

Q385:

```bash
HOST=<GPU machine> WORK=<folder there> HEADS="<run>/heads/nine-1000min <run>/heads/nine" FEATS=<run>/seqs/test.bin \
    ROWS=<run>/seqs/test-rows.jsonl VOCAB=<run>/vocab.npy FLINT_DIR=<run>/flint OUT=out/q385 ALPHA=3 \
    bash docs/data/2026-10-27_cheap_guess/run/q385.sh                                        # then ALPHA=1
```

Licence: the tables and scripts are under this repository's MIT licence ([LICENSE](../../../LICENSE)); the head's
training data and sources are in [docs/drafter.md](../../drafter.md) (M4) and
[bench/staircase_drafter/sources.json](../../../bench/staircase_drafter/sources.json).
