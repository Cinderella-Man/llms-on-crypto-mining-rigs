# Estimate scripts behind the research notes' "(est.)" figures (2026-10-24)

The arithmetic behind several figures the research notes mark "(est.)": short scripts, most with no input files, so
anyone can check or rerun an estimate. They estimate; the measurements that later confirmed or replaced them are in
the registry ([experiments.md](../../../quants/experiments.md)).

| File | What | Used in |
|---|---|---|
| `estimates/payoff.py` | the GPTQ payoff model for K2h's body: Q77's per-group KLD cost of Q8_0 → imatrix Q5_0, ms saved per MB at the Pascal kernels' per-format rates, and which groups a recovery share r lets K2h drop at KLD 0.109 and 0.117 | [gptq_body.md](../../research/notes/gptq_body.md), "Expected gain (est.)" |
| `estimates/payoff2.py` | the same with separate recovery shares for routed experts and always-read tensors (mid and head-like cases) | [gptq_body.md](../../research/notes/gptq_body.md) |
| `estimates/race.py` | Monte Carlo of the shortest of k log-normal reasoning lengths (k = 2–4, CV 0.3–0.8) | [unconventional_ideas.md](../../research/notes/unconventional_ideas.md), first-finish race, "Expected gain (est.)" |
| `estimates/ent.py`, `estimates/ent2.py` | entropy of Q8_0, 6-bit, Q5_0 and Q4_0 codes for blocks of Gaussian, Student-t and Laplace weights (numpy; pure Python) | [unconventional_ideas.md](../../research/notes/unconventional_ideas.md), "Entropy-coded weights" |
| `estimates/udmap.py` | per-layer tensor types and the sliding/global layer pattern of a Flint GGUF (e.g. UD-Q4_K_XL), from its header | [long_context_quality.md](../../research/notes/long_context_quality.md), the global layers' types |
| `estimates/ggufhdr.py` | a pure-Python GGUF header reader, imported by `udmap.py` and by [round11/mix.py](../2026-10-24_q106/run/round11/mix.py) | — |
| `estimates/imat_stats.py` | from an importance matrix: per-expert token counts and, per class and layer band, max channel energy over mean, participation ratio and top-1 % energy share | [gptq_body.md](../../research/notes/gptq_body.md), "Input structure" |

Made on 2026-10-24 on a CPU. Run from `estimates/` (`payoff2.py` reads `payoff.py`; `udmap.py` imports
`ggufhdr.py`):

```bash
cd docs/data/2026-10-24_research_scratch/estimates
python3 payoff.py; python3 payoff2.py; python3 race.py; python3 ent.py; python3 ent2.py
python3 udmap.py flint-UD-Q4_K_XL.gguf                       # reads the header only
LLAMA_SRC=<llama.cpp tree> python3 imat_stats.py imatrix_unsloth.gguf   # Unsloth's Flint importance matrix
```

The scripts are this project's (MIT).
