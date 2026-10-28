# The answer store from agentic data run 1 (2026-10-27)

Data and method of [The answer store on agentic work](../../benchmarks/2026-10-27-answer-store.md): a store of Flint's
own continuations (`--pipe-store`, registry row Q161 in the [registry](../../../quants/experiments.md#registry)) built
from the 1,177 gated agent sessions of agentic data run 1, its provenance and held-out checks, an offline estimate on
held-out sessions and on Q51, and the agent check on the rig with and without it (Miner Llama v2.0, Flint Q8_0, nine
cards, 262,144 tokens): two live arms, and a paired replay of the check's own requests on identical tokens that
measured the store's gain, +1.4 % per generated token (95 % interval 0.4–1.9 %).

| File | What |
|---|---|
| `stores.tsv` | the stores: sessions, turns, tokens, bytes and SHA-256 of the full store (the one to ship) and of the store without the held-out modules' sessions, and the prefixes used for the size curve |
| `chain.tsv` | the full store's provenance chain as [chain.py](../../../bench/agentic_data/chain.py) walked it: 19 gate records from the store back to the workspaces' licence scans, each with its gate, stage, verdict, file count, sources, writers and held-out test files (all PASS, every parent hash and file unchanged) |
| `heldout_check.tsv` | [heldout_check.py](../../../bench/agentic_data/heldout_check.py) on the store's 1,177 sessions (11.9 M words of statements, generated text and tool results): every 8- and 12-word span shared with the agent check's task files or Q51's prompts, and the agent check's exercise names |
| `offline/agentic.md`, `.json` | [store_agentic.py](../../../bench/ngram_replay/store_agentic.py) on the 40 scored held-out sessions (648 requests, 680,600 positions) with the store without their modules: one token ahead per guesser, workload (reasoning, tool call, answer text) and source, for `cont` and the 25 threshold pairs; the pipeline model at the release split, depths 3 and 4 |
| `offline/sizes.md`, `.json` | the same sessions with stores of 1, 2, 4 and 8 M tokens (prefixes of the shuffled training turns) at the engine's thresholds, depth 4 |
| `offline/q51.md`, `.json` | Q51's 38 transcripts with the full store and the store without the held-out sessions: per class (chat, code, edit, agent) and segment, depths 3 and 4 |
| `agent/<arm>/compare.md`, `table.md`, `gonogo.md`, `attempts.csv`, `turns.csv` | the agent check's reports per arm (`store`: the release line with `--pipe-store`; `base`: the release line, run right after it in the same slot), as in [2026-10-26_release_setup](../2026-10-26_release_setup/README.md) |
| `agent/guesses.tsv` | the server's guess counters summed per arm: requests, guesses launched, right, wrong, and guesses that came from the store |
| `replay/replay.md` | the paired replay (2026-10-28): per label (`store`, `base`, and the recorded live run) the cost per prompt and per generated token, the fixed cost per request, cache and time to first token; the ratios to `base` with 95 % intervals over the six traces |
| `replay/requests.tsv` | one row per replayed request and label: trace, request index, prompt tokens processed and cached, prompt time, generated tokens and time, finish reason, tool calls, first-token and end times (the two labels agree on every token count, cache hit, finish reason and tool call) |
| `replay/guesses.tsv` | the server's guess counters summed per replay arm, as `agent/guesses.tsv` |
| `run/build_stores.sh` | both stores, built and gated as `bench/agentic_data/run.sh`'s checks step does, then `chain.py` and `heldout_check.py` |
| `run/replay.sh` | the offline estimate (held-out sessions with the threshold grid, the size curve, Q51) |
| `run/rig_test.sh` | the two agent-check arms on the rig, after copying the store to the rig's internal disk and checking its SHA-256 |
| `run/rig_replay.sh` | the paired test: six recorded traces of the control arm replayed to the release line with and without the store (same prompts, seeds and reply lengths, so both generate the same tokens), through `release_check.sh`'s replay mode |
| `run/replay_dryrun.sh` | the replay mode's dry run without the rig: the client plays the rig with `mock_server.py` at 20 and 25 ms per token, and the report must give 1.25 |

## Commands

```bash
# the stores, their chain and the held-out check (CPU; RUN is bench/staircase_drafter/agentic_head.sh's WORK)
BENCH=$PWD/bench RUN=<head run> Q51=<Q51 prompts.jsonl> WORK=<out> SESSIONS="<run>/s1/sessions <run>/s2/sessions ..." \
  bash docs/data/2026-10-27_answer_store/run/build_stores.sh
# the offline estimate
BENCH=$PWD/bench RUN=<head run> WORK=<out> Q51_GEN=<Q51 gen.jsonl> bash docs/data/2026-10-27_answer_store/run/replay.sh
# the rig: the store arm, then the release line again
ENV_FILE=<release.env> STORE_SRC=<out>/full/agentic.pstore STORE_DIR=<folder on the rig> CLIENT_ROOT=<client folder> \
  OUT=agent bash docs/data/2026-10-27_answer_store/run/rig_test.sh
# the paired replay on the rig, from the control arm's recorded requests (its proxy.jsonl on the client)
ENV_FILE=<release.env> STORE_DIR=<folder on the rig> REPLAY_LOG=<client folder>/out-store-base/proxy.jsonl \
  CLIENT_ROOT=<client folder> OUT=replay bash docs/data/2026-10-27_answer_store/run/rig_replay.sh
# the replay mode's dry run (no rig): the client plays the rig with the mock server
CLIENT=<client> WORK=<scratch folder on the client> REPLAY_LOG=<a proxy.jsonl on the client> \
  bash docs/data/2026-10-27_answer_store/run/replay_dryrun.sh
```

## Provenance and licence

The sessions come from [bench/agentic_data](../../../bench/agentic_data/README.md)'s run 1
([data folder](../2026-10-26_agentic_data/README.md)): the Python 3.12 standard library (PSF-2.0) and Elixir 1.18.4's
library (Apache-2.0), harness texts written by Flint, model output by Flint (Apache-2.0 weights; the makers' NVFP4 GGUF
in vLLM and Unsloth's Q8_0 in llama-server). The store holds the token ids of the text Flint generated, nothing else;
where Flint's edits copy library code, that code keeps its licence. The store itself is not in this repository (124 MB);
its gate records stay on the build machine, and `chain.tsv` summarizes them.
