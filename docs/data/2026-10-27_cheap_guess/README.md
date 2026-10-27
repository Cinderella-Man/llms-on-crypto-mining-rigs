# A cheaper guess of Flint's MTP head (2026-10-27)

The data behind [A cheaper guess](../../drafter.md#a-cheaper-guess-2026-10-27) in docs/drafter.md (section 6, and
section 8, item 0): where one guess of the
staircase head (Flint's MTP head of agentic run 1, the nine-card head) goes on a P106, what a guess may cost on the
agent check before the head stops paying, and how often the head is right at the positions the server's other
guessers leave it; then what the options of Miner Llama's branch `pascal-p106-v3-cheapguess` cost (4.77 ms → 2.93 ms
per guess) and that they compute what the reference computes. `run/replay.sh` measures them on the agent check
(planned). Registry rows Q190–Q194 in [quants/experiments.md](../../../quants/experiments.md) (profile, break-even and
guess order, options, the agent-check replay, the next cuts).

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
| `run/` | the scripts: `build.sh` (the rig build), `profile.sh`, `variants.sh` (rig), `sim.sh` (the offline steps), `agent.sh` (live agent-check arms), `replay.sh` (the paired replay of recorded agent-check traces, release line against the head on the same build), `curate.py` (these tables from the raw outputs) |

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
  (`agent_tokens.py`); the replies hold 129,890 tokens (the server counted 113,440 generated: the template renders tool
  calls a little differently from Flint's own output).
- **Held-out sessions:** the 40 sessions of held-out modules of agentic run 1 and the head's per-position scores from
  `score.py` (inverse-CDF coupling, as served).
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
CLIENT=<client> CLIENT_WORK=<folder there> MODEL=<rig path of flint-Q8_0.gguf> NAME=<tarball name> HEAD=... \
    DRAFT_ENV='...' OUT=out/agent ARMS_LIST="stair base" bash docs/data/2026-10-27_cheap_guess/run/agent.sh
python docs/data/2026-10-27_cheap_guess/run/curate.py docs/data/2026-10-27_cheap_guess \
    --bench v2.0=out/v20/bench.log cheapguess=out/new/bench.log --served release=<server log> nine=... gen=... \
    --sim cost-agent.json cost-agent-spare.json
```

Licence: the tables and scripts are under this repository's MIT licence ([LICENSE](../../../LICENSE)); the head's
training data and sources are in [docs/drafter.md](../../drafter.md) (M4) and
[bench/staircase_drafter/sources.json](../../../bench/staircase_drafter/sources.json).
