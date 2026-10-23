# Host cost of sampling a Flint tool request

llama-server samples a tool request on the host: the request's lazy grammar switches backend sampling off, so each
token's 150,144 logits cross the PCIe link and the sampler chain runs on the CPU. These two benchmarks time that host
side without a GPU. The analysis is in [tool_request_sampling.md](../../docs/research/notes/tool_request_sampling.md)
(registry rows Q55 and Q66–Q69 in [quants/experiments.md](../../quants/experiments.md)).

- [tool_sampling_bench.cpp](tool_sampling_bench.cpp) rebuilds `common_sampler_init`'s chain for a Flint tool request
  (lazy tool grammar, reasoning budget, llama-server's default samplers) on synthetic logits in which an intended token
  wins. It times a sample and accept per token, before and inside a tool call, and the grammar rejection path that runs
  the grammar over the whole vocabulary. It needs only the GGUF's vocabulary and template: `model.gguf
  [n_reasoning_tokens] [dump_grammar]`.
- [llg_bench.py](llg_bench.py) times llguidance's token masks for the same tool-call grammar
  ([flint_tools.gbnf](flint_tools.gbnf), as llama.cpp generates it for a set of tools) on a decoded code sample
  ([stats_code.txt](stats_code.txt)): `python -I llg_bench.py tokenizer.json flint_tools.gbnf`, with `llguidance`
  installed and Flint's `tokenizer.json`.

Build the C++ bench inside a llama.cpp tree (CPU only is enough): copy this folder to `examples/tool-sampling`, add
`add_subdirectory(tool-sampling)` to `examples/CMakeLists.txt`, and build the `llama-tool-sampling-bench` target.
On the rig run it on the Celeron itself (Q68): the workstation figures in the note are scaled by an assumed 4–6×.
