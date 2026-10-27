# Agentic data: coding-agent sessions with a provenance chain

Flint's own agentic sessions, generated so that every one of them can be published: it starts from a task statement
we may publish, works on code we may publish, and every piece of its context traces back to an approved source. The
sessions feed three things: the answer store (`--pipe-store`, Q161), the staircase drafter's training dumps (Q20) and
a dataset on Hugging Face. Why it is built this way, the licence questions and the plan for the long run:
[agentic data pipeline](../../docs/research/notes/agentic_data_pipeline.md).

## What a session is made of

| Part | Where it comes from | Checked by |
|---|---|---|
| Code base | an approved entry of [codebases.json](codebases.json), pinned to an exact version | `gate.py workspaces` |
| Files in the workspace | only files whose [license_scan.py](license_scan.py) verdict is `ok` | `gate.py workspaces` |
| Tests | the code base's own, or written by Flint in a stage-1 session (Flint is Apache-2.0) | `gate.py tasks` |
| The pipeline's change (a bug, a removed body) | [tasks.py](tasks.py), recorded as a diff | `gate.py tasks` |
| System prompt, tool descriptions, task statement templates, the tool runner's messages | written by Flint from the facts in [texts_spec.json](texts_spec.json) (`agent.py texts`, below), or by the owner | `gate.py tasks`, `sessions` |
| Task statement | a template filled with names from the workspace | `gate.py tasks` |
| Tool results | [sandbox/tools.py](sandbox/tools.py) on the workspace, in a container without network; its own lines are texts.json's messages | `gate.py sessions` |
| Model output | Flint (an approved writer in `../staircase_drafter/sources.json`), exact token ids | `gate.py sessions` |

## Pipeline

[run.sh](run.sh) runs every step; each gate writes a record (verdict, hashes of its files, sha256 of its parents'
records) and a manifest, and the next step starts only from passed rows.

1. **Sandbox image** ([sandbox/Dockerfile](sandbox/Dockerfile)): the pinned `python:3.12-slim` without documentation,
   pip, IDLE and turtledemo, plus the tool runner.
2. **Licence scan and workspaces** ([license_scan.py](license_scan.py), [workspace.py](workspace.py)): per file SPDX
   tags, licence-text signatures, copyright lines, vendored folders and folders with a licence of their own;
   workspaces from `ok` files only, with a manifest of every file's origin and hash.
3. **Server**: llama-server with Flint Q8_0 (or a vLLM server with the NVFP4 GGUF, `API=vllm`).
4. **Texts** (`agent.py texts`): Flint writes the system prompts, tool and parameter descriptions, task templates,
   change notes and the tool runner's messages. Each request is a short instruction and the facts as JSON (names,
   parameters, limits, placeholders, values of at most four words), so it holds no wording to copy. A text passes
   when it shares no run of 5 words with anything the pipeline's authors wrote (the instructions and facts,
   `texts_spec_v1.json`, the tool runner's default messages), Flint judges it consistent with its facts, and its
   placeholders, length and the held-out rule check out; Flint rewords a text that failed only by copying.
5. **Stage 1** (`tasks.py write-tests`, `agent.py run`): write unit tests for a module. Verified: module unchanged,
   test file present, all tests pass.
6. **Stage 2** (`tasks.py from-sessions`, `agent.py run`): on each passing stage-1 test file, single-token bugs
   (comparison, arithmetic, boolean operators, small constants, `True`/`False`, a dropped `not`) and removed function
   bodies that make some tests fail. Verified: the test file unchanged, all tests pass.
7. **Gate** over all sessions, then **exports** ([export.py](export.py)): `train/turns.jsonl` (store),
   `train/sessions_seq.jsonl` (dump: one sequence per session plus its generated spans), `hf/` (the dataset: data,
   LICENSE, LICENSES/, NOTICE, README.md), each gated again (`answers`, `dataset`).
8. **Checks**: the store builds from the turns (`datastore_corpus.py`, `datastore build`) and `feats.py seqs`
   reads the session sequences.

```bash
WORK=<out> MODEL=<flint-Q8_0.gguf> HF_DIR=<RecursiveSI/flint small files> Q51=<Q51 prompts.jsonl> \
LLAMA_SERVER=<llama-server> MOUNTS="-v <dir>:<dir>:ro ..." GPUS=3,4 bash bench/agentic_data/run.sh
```

## Elixir

The same pipeline runs on Elixir 1.18.4's standard library (Apache-2.0) in the Elixir sandbox
([sandbox/Dockerfile.elixir](sandbox/Dockerfile.elixir): Erlang/OTP 27 and Elixir copied from the official image into
the Python sandbox's image, `mix test` offline, Mix's build folder in `/tmp`). `workspace.py elixir` makes one Mix
project per module: the module renamed into the `Stdmods` namespace (in code and doctests only) with Flint's change
note as its first line, and `mix new`'s own project files, its test file reduced to the line that runs the module's
doctests. Stage 1 asks for ExUnit tests (five beyond the doctests); stage 2's bugs and removed bodies come from
Elixir's parser (`sandbox/ex_tool.exs`), and its check also requires the module's documentation unchanged. Task rows
carry `lang` and `image`; the texts carry Elixir's own `run_tests` description and templates (`texts_spec.json`,
`langs.elixir`). Steps `ex-image` and `ex-workspaces` (after `texts`) with `EX_MODULES` and `EX_NOTICE`.

## The agent loop

- **Context as token ids.** The first prompt is Flint's chat template (the Hugging Face `chat_template.jinja`,
  rendered with jinja2 as transformers renders it, and compared with llama-server's `/apply-template`), tokenized by
  the server. Every later turn appends the sampled ids and the tool results rendered by the same template
  (`<|observation|><tool_response>…</tool_response>…<|assistant|>`). A turn that calls tools ends with
  `<|observation|>`, which is also the first token of the results, so the session is one append-only sequence.
- **Parsing:** `<think>…</think>` is the reasoning (kept in the history, as Flint's makers ask);
  `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>…</tool_call>` after it are the calls. Calls inside
  the reasoning are ignored and counted.
- **Tools, two profiles** (`agent.py run --tools agx|pi|mixed`, recorded per session):
  - `agx`: `list_dir`, `read_file`, `search`, `edit_file`, `write_file`, `run_tests`, `run_command`;
  - `pi`: the names and parameters of the pi agent's four default tools, `read(path, offset, limit)`,
    `write(path, content)`, `edit(path, edits=[{oldText, newText}])`, `bash(command, timeout)`, so that the store and
    the drafter learn the tool calls of the harness the rig is measured with (the store's tool-call gain came only
    from answers that used the same tools, Q161).

  Descriptions come from `texts.json` either way. Paths outside `/work` (and `/tmp`) are refused; outputs are cut at
  20,000 characters (60 % head, 40 % tail).
- **Sandbox:** one container per session, `--network none`, read-only root, private `/tmp`, 2 GB memory, 512
  processes, `--cpu-shares 256`, label `pascal-quants=1`.
- **Limits:** 24 turns, 20,000 generated tokens per turn (Flint reasons for more than 8,000 tokens before writing a
  test file), the server's context per slot, 30 minutes, the same calls four turns in a row ends the session;
  sampling as the makers recommend (temperature 1.0, top-k 20, top-p 0.95).

## Session format (`sessions/<task_id>.json`)

`task` (the gated task row), `writer` (id, file name, bytes, sha256), `server`, `sampling`, `texts` (path, sha256,
system prompt id), `tools` (the schema), `messages_start`, `prompt_tokens`, `template_matches_server`, `turns` (per
turn: `context_len`, `seed`, `gen_tokens`, `stop`, `stop_id`, `text`, `reasoning`, `content`, `calls` with arguments
and results, `chunk_tokens`, `stop_matches_chunk`, timings), `end`, `verify`, `final_files` (created or changed files
with content), `sequence_len`, `gen_ranges`, `seconds`, `limits`.

## Files

| File | What it does |
|---|---|
| [license_scan.py](license_scan.py) | per-file licence scan of a tree: verdicts and an allow-list |
| [codebases.json](codebases.json) | code bases and environments: origin, version, licence, attribution, status |
| [workspace.py](workspace.py) | workspaces from scanned files, with manifests |
| [texts_spec.json](texts_spec.json) | the facts and instructions Flint writes the harness texts from (version 2; `texts_spec_v1.json` is the first, prose version, kept for the copy check) |
| [texts_report.py](texts_report.py) | a review table of a texts.json: each text, how it was written, its longest run shared with authored text, Flint's judge note |
| [tasks.py](tasks.py) | stage-1 and stage-2 tasks, validated in the sandbox |
| [agent.py](agent.py) | `texts` and `run`: the agent loop against llama-server or vLLM |
| [sandbox/](sandbox/) | the sandbox images (Python; Elixir in `Dockerfile.elixir`), the tool runner, and `ex_tool.exs`, the Elixir parser helper (renames, mutations, stubs, documentation hashes) |
| [notices/](notices/) | NOTICE files of approved code bases that ask for one (Elixir 1.18.4's) |
| [gate.py](gate.py) | the gates: workspaces, tasks, sessions, answers, dataset |
| [export.py](export.py) | store, dump and dataset exports; several runs (writers) export as one, with one id range |
| [heldout.py](heldout.py) | the drafter's held-out sessions: every session of every 20th module (by name hash), never trained on |
| [heldout_check.py](heldout_check.py) | an independent check that sessions hold no text of the project's own tests: 8- and 12-word spans of the agent check's task files and Q51's prompts, and the agent check's exercise names, in generated text, tool results and statements |
| [agx.py](agx.py) | shared helpers: hashes, the 12-word rule, licence markers, the template, the sandbox |
| [run.sh](run.sh) | the whole pipeline in one command |
