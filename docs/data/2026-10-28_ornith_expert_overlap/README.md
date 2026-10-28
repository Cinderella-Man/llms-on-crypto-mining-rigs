# Ornith-1.5's expert overlap for 1–8 consecutive tokens (registry Q411): planned, not run yet

How many distinct experts per MoE layer 1 to 8 consecutive tokens of Ornith-1.5 touch: the cost of a draft-and-verify
step that reads each distinct expert once ([2026-10-28_multitoken_kernels](../2026-10-28_multitoken_kernels/README.md),
Q211), and the u(m) of [cheap_mtp_draft.md](../../research/notes/cheap_mtp_draft.md) §3, which assumes 15 / 22 / 28.5
at 2 / 3 / 4 tokens (independent routing over 256 experts with 8 used: 15.75 / 23.3 / 30.5). Flint's answer is Q51
([bench/expert_overlap](../../../bench/expert_overlap/README.md)): 80 % of random at 8 tokens.

**Status (2026-10-28): the tools are ready; nothing has run.**

- `bench/expert_overlap/route_dump.cpp` reads a plain text (`ROUTE_TEXT`, cut into `ROUTE_CHUNK`-token sequences) as
  well as token sequences, and records the router's experts per position and MoE layer.
- `bench/expert_overlap/analyze.py` counts the distinct experts in every window of 1–8 and 16 tokens, against
  independent random routing and against the same tokens at shuffled positions (the layer's own expert popularity
  without locality); `-` in place of GEN.jsonl for ROUTE_TEXT dumps.
- `run/build_route_dump.sh` builds the dump tool against an existing llama.cpp CUDA build; `run/run_overlap.sh` runs one
  model on one text in Docker and the analysis after it.

Planned runs: Ornith-1.5-35B Q8_0 and Ornith-1.5-122B UD-Q2_K_XL, teacher-forced over the held-out agent sessions
(12 × 8,192 tokens) and the Exercism code (80 × 512) of [2026-10-28_ornith_mtp_accept](../2026-10-28_ornith_mtp_accept/README.md):

```bash
D=docs/data/2026-10-28_ornith_expert_overlap/run
LLAMA=<llama.cpp checkout> BUILD=build-cuda SRC=bench/expert_overlap OUT=<work>/bin/llama-route-dump bash $D/build_route_dump.sh
BIN=<work>/bin/llama-route-dump ANALYZE=bench/expert_overlap/analyze.py GPUS=<two GPUs, ~48 GB> \
  MODEL=<Ornith-1.5-35B-Q8_0.gguf> TEXT=agent-heldout.txt CHUNK=8192 CHUNKS=12 OUT=<work>/35b-agent bash $D/run_overlap.sh
# the same with TEXT=code-exercism.txt CHUNK=512 CHUNKS=80, then MODEL=<Ornith-1.5-122B-UD-Q2_K_XL.gguf>
```
