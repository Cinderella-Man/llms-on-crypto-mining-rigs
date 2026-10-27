# Q123: where Flint stops reasoning, per quant

Does a quantized file change where Flint ends its reasoning? At every `</think>` that closes a reasoning block and at
every hesitation marker ("maybe", "but", "actually", "wait" and the like) in Q106's agent conversations,
[stop_decision.py](../../../bench/long_context_kld/stop_decision.py) compares each file's log-probability of the actual
token with UD-Q4_K_XL's. The outcome, closed as a reasoning-length lever, is in the registry row Q123
([quants/experiments.md](../../../quants/experiments.md)), in the "Where reasoning stops (Q123)" section of
[bench/long_context_kld/README.md](../../../bench/long_context_kld/README.md) and in
[answer_time.md](../../research/notes/answer_time.md). These are the full tables behind them, with the split by the
model that produced each conversation.

- **24 chunks, Q8_0 as the dumps' reference** (`stop-agent.md`): 230 stops and 842 markers. Every file is within
  0.008 nats of UD-Q4_K_XL at stops and 0.011 at markers, every interval covering 0, against the 0.05-nat kill line.
  Every file gives the actual `</think>` 0.886–0.894 probability.
- **12 chunks, BF16 as the reference** (`stop-bf16agent12.md`): 132 stops, 663 markers; all files within 0.017 nats.
  Only K2hg's markers exclude 0 (−0.017, −0.029 to −0.004), towards fewer re-checks.
- **The same 12 chunks cut from the 24-chunk dumps** (`stop-agent-sub12.md`) give each file's own log-probabilities
  unchanged, so the BF16 half differs from the full run only by its subset of chunks, not by its reference.

| File | What |
|---|---|
| `stop-agent.md` | 24 chunks (98,280 scored tokens), seven files against UD-Q4_K_XL and Q8_0: stops and markers, by producer, and the kill rule |
| `stop-bf16agent12.md` | 12 chunks (49,140 tokens) with BF16 as the dumps' reference: UD-Q4_K_XL, Q8_0, G3, G3g, K2h, K2hg |
| `stop-agent-sub12.md` | the same 12 chunks from the 24-chunk (Q8_0-referenced) dumps |
| `run/run.sh` | the three analyses |
| `run/regress.sh` | the check that `analyze.py` after its refactor (`load_dumps`, `check_same_tokens`, `piece_decoder`) gives byte-identical tables to the version before it, on the full and the `--only-chunks` path; it did |

Made on 2026-10-24 with `bench/long_context_kld` at commit `5143f10`, on CPU on the 7-GPU quant machine, from the
per-token dumps of Q106 and Q107 (registry rows Q106, Q107): 24 chunks of 8,192 tokens of Flint's BFCL v4 agent
conversations produced by G3 or UD-Q4_K_XL, scored by every file. The vocabulary comes from a Flint GGUF
(G3's). A direct recount of the dumps (token 150007 is `</think>`, 230 rows) gives the same mean shifts: G3 −0.0044,
K2h +0.0066, K1h +0.0080, G3h +0.0054.

```bash
TOOLS=bench/long_context_kld DUMPS=... MODEL=flint-q73-G3.gguf OUT=... PYTHONPATH=/path/to/llama.cpp/gguf-py \
  bash run/run.sh
```

The scored text is Flint's own output in BFCL v4 multi-turn conversations (BFCL: Gorilla project, Apache-2.0; Flint:
Apache-2.0); only per-position statistics are included here.
