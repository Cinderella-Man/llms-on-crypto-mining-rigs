# Publishable agentic data: Flint's coding-agent sessions for the store, the drafter and Hugging Face (2026-10-26)

Scope: the answer store (`--pipe-store`, Q161) and the staircase drafter head (Q20) both learn from Flint's own text,
and both must be rebuilt from data we may publish. The project's main use case is a coding agent, so the text should
be Flint working as an agent: many turns, tool calls, file contents and test output in the context. This note asks:

1. What was built before, which mistakes made all earlier data unusable, and which tools survive.
2. How to guarantee that every agentic session starts from a publishable prompt and works on publishable code: the
   provenance of a session, the gates, and the licence questions (with primary sources).
3. Which code bases to use as workspaces, from what is already on our machines.
4. How to make many diverse tasks that are clean by construction.
5. The session generator, and a transcript format that feeds the store, the drafter's dump and a published dataset.
6. How much data, how fast, on which GPUs.
7. A pilot on the 7-GPU quant machine, and the plan for the long run.

Tools: [bench/agentic_data](../../../bench/agentic_data/README.md). "(est.)" marks this note's own arithmetic.

## Summary

- **Every earlier drafter and store dataset failed for one of two reasons:** a prompt whose author was not who its
  card implied (scraped ChatGPT logs, GPT-written instructions and maths, LLM-written tool sets), or code whose licence
  was decided per folder or per package instead of per file. Checks ran on id lists made by the same code they were
  meant to check, so two filter bugs went through. The agent prompts were single turns with invented tool results,
  2.5 % of the tokens. Edit prompts reused the test set's instructions word for word.
- **Two problems were not on the list.** The "own" prompt texts (system prompts, tool descriptions, task templates)
  were written by an AI coding assistant, and the owner's rule excludes text written by OpenAI, Anthropic or Google
  models; Anthropic's usage policy forbids using outputs to train a model without authorization. And the approved
  OpenR1-Math problems come from NuminaMath 1.5, which mixes in synthetic sources; the approval took the whole split.
- **A session is publishable when each of its six inputs is:** an approved code base pinned to an exact version and
  filtered file by file, an environment that holds nothing else, harness texts written by Flint (Apache-2.0) or by
  the owner, a task statement that is such a text filled with names from the workspace, the pipeline's own change to
  the workspace as a diff, and Flint as the writer. Tool results are then the sandbox's output on those files.
  [bench/agentic_data](../../../bench/agentic_data/README.md) builds exactly that and gates every stage; each gate
  record names its parents by sha256, so every published row chains back to an approved code base, text and writer.
- **Licences:** MIT, BSD, Apache-2.0 and PSF code may appear verbatim and modified inside an Apache-2.0 dataset; each
  keeps its own licence and notices. Ship `LICENSES/<SPDX>.txt`, a NOTICE with every project's attribution and the
  change summaries (PSF and Apache-2.0 ask for them), and per-row licence fields. Scan every file: CPython's library
  has 28 modules with notices of their own, llama.cpp 45 files that are not MIT.
- **Code bases on hand:** CPython 3.12.15's library, llama.cpp at `d77256999` with its tests, fsspec 2026.9.0 and six
  permissive packages with their test suites from local images (SymPy, NetworkX, mpmath, jsonschema, dill, NumPy), all
  approved, with pytest from a local image. Elixir 1.18.4's standard library (246 files, all clean) and Erlang/OTP
  27's sources sit in a local image too, and an Elixir sandbox built from it runs `mix test` offline (both approved 2026-10-26). A list
  of 20 more to download, each needing approval.
- **Tasks:** write tests (stage 1), then repair a single-token bug and implement a removed function, checked against
  those tests (stage 2). Statements are Flint-written templates; Flint never sees the answer.
- **Pilot** (72 minutes on two V100s, Flint Q8_0, eight standard-library modules): 22 sessions, 235k generated
  tokens, every gate passed, the store built from them, the dump's sequence file read them. Stage-1 sessions wrote
  test suites for 4 of 8 modules (18–50 tests each); all 14 stage-2 repairs and implementations on those tests
  passed. It found nine problems, all fixed or planned (section 7): most notably, Flint needs ≥ 32k tokens per turn
  and 64k contexts, and it calls tools by the names it knows (`shell`), so the pi agent's tool names are now a profile.
- **Plan:** section 8 gives the command, the monitoring, the stop rules, the final gate and the approvals needed first.

## 1. What was built before, and what went wrong

### 1.1 What exists

| Piece | What it does | State |
|---|---|---|
| `staircase_drafter/prompts.py` | task texts (edit, agent), OpenR1-Math problems; the first runs' sets as `EXCLUDED_SETS` | texts shared with Q51; written by an AI assistant |
| `staircase_drafter/clean_prompts.py` | "implement this function" stubs, whole-file edits and single agent turns on the standard library, llama.cpp and permissive packages; oasst2 chat | no `source_id`/`row_id`, so the gate cannot read its output |
| `staircase_drafter/tool_prompts.py` | ToolACE and glaive tool prompts | excluded sources |
| `staircase_drafter/sources.json`, `check_sources.py` | approved sources and a stage gate with hashed parent records | drafted 2026-10-25; its `derived` stage now records the pilot's store; the `prompts` stage was never run, see 1.3 |
| `staircase_drafter/generate.py`, `vllm/` | answers with exact token ids from llama-server or vLLM (NVFP4 on two RTX 3090s, ~2.4–2.7k tok/s) | reusable |
| `staircase_dump/`, `feats.py`, `feed_split.py`, `train.py`, `score.py` | the dump (hidden states, Q8_0 top-20), packs, training, scoring | reusable |
| `ngram_replay/datastore_corpus.py`, `datastore.cpp` | the answer store from generated answers | reusable |
| `agent_check/` | the pi coding agent on Exercism tasks in Docker sandboxes, with a logging proxy | the evaluation of agentic speed; its tasks are held out |

### 1.2 Mistakes and their root causes

| Mistake | Evidence | Root cause | In the new pipeline |
|---|---|---|---|
| Prompts from ShareGPT (scraped ChatGPT logs with an uploader's "apache-2.0" tag), UltraChat and both Magicoder sets (GPT-written) | 29,523 answers, 44.4 M of 74.7 M answer tokens (M3 table, [drafter](../../drafter.md)) | the card's licence was read as permission; nobody asked who wrote the text | sources name their writer; only Flint and the owner write texts |
| NuminaMath-CoT: ~45 % GPT-4-written problems | `synthetic_math`, `synthetic_amc`, `orca_math` | same | not used |
| ToolACE, glaive-function-calling-v2, Hermes FC | written by models that are not named | same | not used |
| OpenR1-Math-220k approved as a whole | its `problem` field comes from NuminaMath 1.5, which still has a `synthetic_math` source; contest and forum problem texts may not be the publisher's to license | the approval did not look one level up the data's lineage | open: filter by `source`, and the owner decides on contest texts (section 8) |
| Code inputs under other licences | tqdm and pip's certifi (MPL-2.0), resolvelib (ISC), llama.cpp `vendor/` and build outputs | licence decided per package or folder by name; a path-prefix filter (`startswith`) let `build-cuda-hd/` and `vendor/` through | per-file scan ([license_scan.py](../../../bench/agentic_data/license_scan.py)); see 2.4 |
| Test contamination | `prompts.py`'s edit instructions are Q51's; 1,544–1,585 answers shared 12-word spans with Q51 | test and training code shared constants | held-out spans from Q51 and the agent check's tasks; every text field is checked |
| Checks on lists, not on data | a Magicoder set was grouped as "code" once; the `startswith` bug | the check read id lists derived by the code under test | gates recompute from the files: hashes against the scan, statements re-filled from templates, token chains rebuilt |
| No provenance per row | rows carried a file's base name (`source`), so stdlib and llama.cpp names could collide; `clean_prompts.py` never wrote `source_id`/`row_id`, which `check_sources.py` requires | the generator and the gate were written apart and never run together | one manifest per workspace, task and session; the gate runs in the same script as the generator |
| "Agent" data that was not agentic | single turns: a system prompt and a task, or one `read_file` with the file pasted as its result; 1,100 of 7,150 prompts, 2.5 % of tokens | no sandbox, no tool execution | real multi-turn sessions in sandboxes |
| The pi harness for data | pi declares MIT in its package.json but ships no licence file; its system prompt and tool texts would be in every transcript, and their authorship is unknown | the evaluation harness was the only agent at hand | our own loop; pi stays the evaluation harness only |
| "Our own" texts written by an AI assistant | the system prompts, tool descriptions and task templates in `prompts.py`, `clean_prompts.py` and the Q51 builder were drafted by Claude sessions; `sources.json` lists them as "this repository's authors" | the owner's rule (not OpenAI/Anthropic/Google-generated) was applied to datasets, not to our own files | Flint writes the harness texts (`agent.py texts`); the owner may replace any |
| A loose writer rule | `sources.json` admits any model file named `flint-*.gguf` | glob instead of identity | writers by file name and size (and sha256 when known); the gate checks the session's recorded file |

Old data: the earlier work folders of Q20 on the quant machine's NVMe and both bulk disks are gone (the owner's
deletion); the folder of the stopped 2026-10-25 restart (prompts built by the old `clean_prompts.py`, and cached
OpenR1/oasst2 shards) remains. Nothing of it was read for this note.

### 1.3 What can be reused

| As it is | With fixes | Not at all |
|---|---|---|
| `generate.py --api vllm` and `vllm/` (serving; add a prefix-cache switch), `staircase_dump/`, `feats.py seqs`, `feed_split.py`, `datastore_corpus.py` and `datastore.cpp`, the training and scoring code, `check_sources.py`'s `dump` and `derived` stages (they chain from our `answers` record; `derived` gained `--tool` for programs such as the datastore), the agent check's Docker sandbox pattern and task validation idea | `sources.json` (add code bases, text writers, OpenR1's source filter, writer identity), `check_sources.py`'s `prompts` stage (per-file verdicts, text writers), `train.py` (train only inside `gen_ranges` when it reads whole-session sequences), `clean_prompts.py` for single-turn chat and maths (`source_id`, `row_id`, Flint-written task texts) | the task and agent texts of `prompts.py` (Q51's, AI-assistant-written), `tool_prompts.py`, `EXCLUDED_SETS`, pi as a data harness, the stopped restart's prompt file |

## 2. Provenance of a session

### 2.1 What a session is made of

| Input | Requirement | Recorded as |
|---|---|---|
| Code base | an approved entry of [codebases.json](../../../bench/agentic_data/codebases.json): exact version (commit, image digest or wheel), licence (SPDX), attribution, the approved source in `sources.json` | `codebase_id`, `source_id`, `licence` |
| Workspace | only files whose scan verdict is `ok`; licence file kept; every file's origin path and sha256 | workspace manifest (`workspace.py`) |
| Environment | the sandbox image holds only approved components (CPython 3.12.15 and our tool runner for the pilot); documentation, pip, IDLE removed | image id in `codebases.json` |
| Harness texts | system prompts, tool and parameter descriptions, statement templates written by Flint from [texts_spec.json](../../../bench/agentic_data/texts_spec.json) (or by the owner); tool names and parameters are fixed identifiers | `texts.json` (writer, seed, hash of the generated tokens per text) |
| Task | a template filled with names from the workspace (paths, test names, a function name); the pipeline's change as a diff; the check that decides success | task row (`tasks.py`) |
| Writer | an approved Flint file, by name and size (and sha256 when known), with the engine build and sampling | `writer`, `server`, `sampling` |
| Outputs | Flint's token ids; tool results from the sandbox on the workspace; the final files | session file (`agent.py run`) |

The session file is the per-session manifest: the task row (with its workspace manifest's hash), the writer, the
engine, the texts file's hash and the system prompt's id, the tool schema, the first messages, the first prompt's
token ids, every turn (context length, seed, generated ids, the parsed reasoning, content and calls, each call's
arguments, output, exit code, files read and written, the tool-result token ids), the end reason, the check's result,
the changed files, and the generated spans of the whole sequence. The dataset manifest is the chain of gate records:
each holds its files' hashes, `sources.json`'s and `codebases.json`'s hashes, the held-out files' hashes, its
parents' record hashes, a verdict, and a TSV with one row per item.

### 2.2 Gates

| Stage | Checks ([gate.py](../../../bench/agentic_data/gate.py)) |
|---|---|
| Code base and workspaces | the code base is `approved` and its source is in `sources.json`; every code or licence file equals the scanned file (sha256) with verdict `ok`; scaffold files are empty; the folder equals its manifest; no file shares a 12-word span with a held-out test (Q51's prompts, the agent check's tasks) |
| Texts | the writer is approved; every text passed its own checks when written (placeholders exactly once, length, about the requested tool or parameter, held-out rule) |
| Tasks | the workspace is gated (stage 2: its base workspace is, its test file is a passed session's file, byte for byte); the statement is the template filled with the placeholders; every placeholder names something that is in the workspace or is the file to be created; held-out rule |
| Sessions | the task is a passed row, unchanged; system prompt, tool schema and first messages are `texts.json`'s; the writer's file is approved; the chat template rendered as the server renders it; the token chain is append-only and matches the turns; every tool call ran in the sandbox on paths inside the workspace; no tool output or generated text has a licence or copyright line that is not in the workspace's files or the session's own writes, nor a path outside the workspace and CPython's library; held-out rule on every output. A broken rule fails the gate; a content finding excludes the session and the rest go on |
| Store and dump exports | every row is a passed session's tokens (or its conversation, for the turn prompts), exactly; the record has `check_sources.py`'s layout, so its `dump` and `derived` stages chain from it: the pilot recorded its store corpus and its store that way |
| Published dataset | every row is a passed session with its code base's licence; `LICENSE`, `NOTICE`, `README.md` with a YAML header and `LICENSES/<id>.txt` for every licence are present; no machine paths; held-out rule |

### 2.3 What the published dataset ships

| File | Content |
|---|---|
| `data/sessions.jsonl` | per session: messages (system, user, assistant with `reasoning_content` and `tool_calls`, tool), tools, task (type, statement, template id, the pipeline's diff, the check), code base, source, licence (SPDX), workspace files (path, sha256, origin), writer, engine, sampling, end reason, verified |
| `data/tokens.jsonl` | per session: the exact token ids and the generated spans |
| `LICENSE` | Apache-2.0, for the dataset's own parts |
| `LICENSES/` | the full licence text of every code base used (REUSE layout: `LICENSES/MIT.txt`, `LICENSES/Python-2.0.txt`, …) |
| `NOTICE` | attribution per code base (copyright line, licence, origin and version), Flint's attribution, the change summaries, and that the projects do not endorse the dataset |
| `README.md` | the card: YAML header with `license: [apache-2.0, mit, bsd-3-clause, other]` and `license_name: python-2.0`, `license_link: LICENSES/Python-2.0.txt` when CPython is in; provenance, fields, limits |

### 2.4 The licence questions

**Can MIT, BSD, Apache-2.0 and PSF code appear verbatim, and modified, inside an Apache-2.0 dataset?** Yes. All four
grant redistribution and modification; each attaches notice conditions, and the code keeps its own licence. The
Apache Software Foundation's third-party policy puts MIT, BSD-2/3-Clause, the PSF licence and Apache-2.0 in its
Category A, licences whose code may be included in Apache-2.0 products ([ASF resolved](https://www.apache.org/legal/resolved.html)).
The dataset's Apache-2.0 then covers our own parts only; the card lists every licence.

| Licence | What redistribution requires (primary text) | What we do |
|---|---|---|
| MIT ([OSI](https://opensource.org/license/mit)) | the copyright notice and the permission notice "shall be included in all copies or substantial portions" | `LICENSES/MIT.txt` with the project's copyright line; files keep their headers |
| BSD-2/3-Clause ([OSI 2](https://opensource.org/license/bsd-2-clause), [OSI 3](https://opensource.org/license/bsd-3-clause)) | source redistributions "must retain the above copyright notice", the conditions and the disclaimer; BSD-3: no use of the names to endorse | the licence text with each project's notice; the card claims no endorsement |
| Apache-2.0 ([text](https://www.apache.org/licenses/LICENSE-2.0)) | §4(a) a copy of the licence; §4(b) modified files "carry prominent notices stating that You changed the files"; §4(c) keep copyright, patent, trademark and attribution notices; §4(d) the NOTICE file's attribution notices | the licence text; the NOTICE carries any upstream NOTICE; every row records the pipeline's diff and the model's changes are in its messages |
| PSF ([Python 3 licence](https://docs.python.org/3/license.html)) | PSF's licence agreement and copyright notice retained; a derivative work must "include … a brief summary of the changes made to Python" | CPython's whole licence file as `LICENSES/Python-2.0.txt` (SPDX `Python-2.0` is the whole stack, `PSF-2.0` only its first part); the change summary in NOTICE and per row |

**Files under another licence inside a permissive repository.** Scan every file, not the folder: SPDX short
identifiers (`SPDX-License-Identifier:`, placed "at or near the top of the file in a comment",
[SPDX 2.3 Annex](https://spdx.github.io/spdx-spec/v2.3.1/using-SPDX-short-identifiers-in-source-files/)), REUSE
`.license` files and `LICENSES/` folders ([REUSE 3.3](https://reuse.software/spec-3.3/)), the signature sentences of
licence texts, copyright lines, folders with a licence file of their own, and vendored folders. `license_scan.py`
does this with the standard library; [ScanCode Toolkit](https://github.com/aboutcode-org/scancode-toolkit) is the
thorough second opinion (an install, so the owner's call). What it found:

| Tree | Files | `ok` | Other licences found |
|---|---|---|---|
| CPython 3.12.15 `Lib/` | 731 | 686 | HPND-style notices (http.cookies, trace, uu, logging, xml.etree, xmlrpc.client, platform; 12 files), MIT (tomllib, 4 files; tarfile), Apache-2.0 (profile, pstats), Apache-2.0 + MIT (three asyncio files, from uvloop), Zlib (sqlite3, turtle), BSD (optparse), GPL-2.0 or PSF (urllib/robotparser.py) |
| llama.cpp `d77256999` | 3,880 | 3,708 | Apache-2.0 WITH LLVM-exception (34 SYCL files, Intel), Apache-2.0 (OpenVINO back-end, the Android gradlew, two model chat templates), Unlicense (`common/base64.hpp`), an LGPL mention (an Android build file); `models/templates/` holds other models' chat templates and is left out whole; 37 vendored, 52 binary, 38 over 256 KB |
| SymPy 1.14.0 (wheel) | 1,561 | 1,526 | 14 vendored, 14 in folders with their own licence, 1 MIT file |
| NetworkX 3.7 (wheel) | 598 | 581 | one Creative Commons notice, one "proprietary" word |
| NumPy 2.2.6 (wheel) | 868 | 783 | 50 in folders with their own licence |

The conservative rule, used here: a workspace holds only `ok` files (no notice, or the tree's own licence); anything
else waits for a decision per file. Permissive exceptions (a Zlib or MIT file in a PSF tree) can come back as
separately attributed entries.

**Test fixtures and data files.** They fall under the repository's licence unless they say otherwise, but they are
where third-party material hides: copied web pages, corpora, other projects' test suites, Unicode tables, images.
Binary files, files over 256 KB, files with a foreign notice and folders with their own licence stay out of the
workspace; the agent can only read what is in it.

**Tracebacks and output quoting CPython or third-party code.** A traceback quotes file paths and single source
lines. In the sandbox the only code outside the workspace is CPython's library and our tool runner, both attributed,
so whatever a traceback quotes is covered. The gate excludes a session whose output names a path outside the workspace
and CPython (the interpreter's and the shell's own messages excepted), or carries a licence or copyright line found in
none of the workspace's files. The GNU project's FAQ holds that a program's licence does not cover its output unless
the output copies parts of the program ([GPL FAQ](https://www.gnu.org/licenses/gpl-faq.html#GPLOutput); not
re-fetched for this note); help texts and man pages are such parts, so the sandbox has no documentation and refuses
paths outside the project.

**Model output.** Flint's weights are Apache-2.0 and its card sets no terms on outputs ("Hosted API terms and data
practices are separate from the weight license"). The US Copyright Office holds that material whose expressive
elements a machine produced lacks human authorship ([88 FR 16190](https://www.govinfo.gov/content/pkg/FR-2023-03-16/html/2023-05321.htm);
[Part 2 report, 2025](https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-2-Copyrightability-Report.pdf)),
so Flint's text is probably free of copyright, and labelling it Apache-2.0 is harmless. Where Flint copies a
workspace file, that part stays under the file's licence. A model can also reproduce memorized code; the gate excludes
a session with any generated licence or copyright line that is not in the workspace, which catches the usual sign of
it.

**Text written by other vendors' models.** Anthropic's usage policy (the version effective 2026-11-12) forbids using
outputs "to train an AI model" without prior authorization ([AUP](https://www.anthropic.com/legal/aup)); its
commercial terms forbid using the services "to train competing AI models"
([terms](https://www.anthropic.com/legal/commercial-terms)); OpenAI's terms forbid using output "to develop models
that compete with OpenAI", and Google's Gemini API terms the same. Whether templates drafted by an assistant reach
that far is unsettled; the conservative choice is the one taken: harness texts from Flint or the owner, never from
those models. The pipeline's code is not data and is not affected; its few functional messages ("error: … does not
exist", "[exit code 1]") are listed in `sandbox/tools.py` for the owner to keep or reword.

**Issue and pull-request texts.** GitHub's terms license content added to a repository under the repository's licence
(D.6) but treat issues and comments as content viewable within the service (D.5)
([GitHub ToS](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service)). So a repository's MIT
licence does not clearly cover its issue texts, and SWE-bench-style statements are out; statements are ours.

**Licence metadata on Hugging Face.** The card's YAML `license` field takes Hub identifiers (`apache-2.0`, `mit`,
`bsd-3-clause`, …, `other`); with `other`, `license_name` and `license_link` name the text
([dataset cards](https://huggingface.co/docs/hub/datasets-cards), [licences](https://huggingface.co/docs/hub/repositories-licenses)).
There is no Python identifier, so CPython goes in as `other` with `license_name: python-2.0`; `huggingface_hub`
accepts a list of licences.

## 3. Code bases

### 3.1 Already on our machines (no download)

| Code base | Where | Licence | Tests | Status |
|---|---|---|---|---|
| CPython 3.12.15 `Lib/` | the pinned `python:3.12-slim` image | Python-2.0 | none (the image removes `Lib/test`); Flint writes them (stage 1) | approved (python-stdlib); 686 `ok` files, single modules |
| llama.cpp `d77256999` | the quant machine's checkout (`git archive`) | MIT | `tests/` (C++, needs a CPU build), `gguf-py/tests` (pytest, NumPy) | approved outside `vendor/` and build folders, test folders included |
| fsspec 2026.9.0 | the vLLM image's site-packages | BSD-3-Clause | `fsspec/tests` (pytest) | approved (pypi:fsspec 2026.9.0) with its tests |
| SymPy 1.14.0 | the vLLM image | BSD-3-Clause | 65 test folders; its own runner works without pytest | approved with its tests |
| NetworkX 3.7 | the vLLM image | BSD-3-Clause | 25 test folders (pytest) | approved with its tests |
| mpmath 1.3.0 | the vLLM image | BSD-3-Clause | `mpmath/tests` | approved with its tests |
| jsonschema 4.26.0, referencing 0.37.0 | the vLLM image | MIT | unittest-style suites | approved with its tests |
| dill 0.4.1 | the vLLM image | BSD-3-Clause | `dill/tests` | approved with its tests |
| NumPy 2.2.6 | the vLLM image | BSD-3-Clause (bundled parts excluded by folder) | 14 test folders (pytest) | approved with its tests |
| pytest 8.4.2, pluggy, iniconfig, pygments | the agent check's image | MIT, BSD-2-Clause | – | approved as the `py312-pytest` environment |
| Elixir 1.18.4 standard library (`elixir`, `eex`, `ex_unit`, `logger`, `iex`, `mix`; 246 files, 145k lines) | `/usr/local/src/elixir/lib/*/lib` of the official `elixir:1.18.4-otp-27` image | Apache-2.0 | none (the image removes them); the modules' doctests, and ExUnit tests written by Flint (stage 1) | approved (elixir-stdlib, 2026-10-26); 246 of 246 files `ok` |
| Erlang/OTP 27.3.4.18 library sources (`.erl`, `.hrl`) | the same image | Apache-2.0 | none in the image | proposed after Elixir; 1,445 of 1,465 files `ok` (16 large generated files, 3 with a "proprietary" marker, `leex.erl` with a BSD notice left out) |
| Erlang/OTP 27.3.4.18 and Elixir 1.18.4 runtimes | the same image, copied into the Python sandbox | Apache-2.0 | – | approved as the `ex118-sandbox` environment (2026-10-26) |
| This repository | the laptop | MIT | few | not proposed: it holds the held-out tests |

Not usable: the Exercism tasks of `bench/agent_check` (MIT) are the agentic-speed evaluation, and Exercism's problem
specifications are shared by every language track, so no Exercism track may be a workspace. The other projects'
images on the quant machine belong to its owner's other work, and some of them are AGPL.

**Elixir.** Coding agents are the rig's main use and Elixir is a language its users work in, so Elixir sessions are
the next language after Python. The official `elixir:1.18.4-otp-27` image is already on the quant machine; it keeps
Elixir's library sources and removes their tests. [sandbox/Dockerfile.elixir](../../../bench/agentic_data/sandbox/Dockerfile.elixir)
copies its Erlang/OTP and Elixir runtimes into the Python sandbox's pinned image (which already has the libraries the
runtime links), so the tool runner stays the same and `mix test` runs with no network. Checked: a Mix project holding
`OptionParser`'s file, renamed `Stdmods.OptionParser`, compiles and passes its 35 doctests through `tools.py`'s
`run_tests`, on a read-only root.
- **Every byte of a session must be publishable:** code and documentation from the approved library, project files
  from Elixir's own generator, wording from Flint (or the owner), the pipeline's changes as mechanical one-token
  diffs. No text written by another vendor's model enters a workspace or a statement.
- **Workspaces** (`workspace.py elixir`): one standard-library module per Mix project, renamed into a `Stdmods`
  namespace (a copy named `OptionParser` would replace the loaded module that Mix itself uses). The rename touches
  module names in code (found by Elixir's parser) and in doctests, nothing else; the gate checks that removing the
  prefix gives back the scanned file byte for byte. The file's first line is a comment saying it was changed
  (Apache-2.0 asks for that in changed files): a one-sentence note Flint writes once, like the statement templates.
  The project around it is `mix new stdmods --module Stdmods` run in the sandbox: `mix.exs`, `.formatter.exs` and
  `test/test_helper.exs` unchanged, its `test/stdmods_test.exs` reduced to the line that runs the module's
  doctests. Elixir's NOTICE and the Apache-2.0 text sit beside them. Candidates are modules with public functions and doctests:
  `OptionParser` (37 doctest lines), `Version` (22), `URI` (53), `Keyword` (137), `Range` (75), `Base` (49),
  `Calendar.ISO` (159), `Duration` (44), and EEx's compiler with tests from stage 1; not the compiler's own modules
  (`Kernel`, `Macro`, `Code`).
- **Tests:** the module's doctests (written by the Elixir team in its documentation), plus ExUnit tests Flint
  writes in stage 1, as for CPython's modules. Stage 1 asks for at least five tests beyond the doctests.
- **Changes:** the same single-token mutations (`<`/`<=`, `==`/`!=`, `and`/`or`, `+`/`-`, small integers,
  `true`/`false`, a dropped `not`/`!`) and removed bodies (`raise "not implemented"`), found with Elixir's own parser
  (`Code.string_to_quoted` with column and token metadata; `sandbox/ex_tool.exs`) and kept when 1 to 30 % of the
  tests fail. After a stage-2 session the test files, the project files and the module's documentation (a hash of
  every `@moduledoc`, `@doc` and `@typedoc`) must be unchanged, so a fix cannot edit a doctest instead of the code.
- **Texts:** Elixir's own `run_tests` description (`mix test`), its write-tests, fix-bug and implement templates and
  the change note, written by Flint from `texts_spec.json`'s `langs.elixir` requests (`agent.py texts --extend`);
  the system prompts and the other tool descriptions are shared with Python.
- **Held out:** projects used by the Elixir agent benchmarks we may measure with (Livebook, Plausible Analytics,
  GenStage, the Nimble libraries) and their task texts; the 12-word rule extends to those texts.
- **Erlang:** OTP's `.erl` sources first as read-heavy tasks (explain, investigate), later with EUnit.

### 3.2 Worth adding (each a download, each needs the owner)

Chosen for small, dependency-free test suites that run offline, permissive licences, and variety of domain:

| Project | Licence | Language | Tests | Why |
|---|---|---|---|---|
| more-itertools | MIT | Python | unittest, no dependencies | many small documented functions |
| toolz | BSD-3-Clause | Python | pytest, no dependencies | functional utilities |
| boltons | BSD-3-Clause | Python | pytest, no dependencies | 30+ independent modules |
| tabulate | MIT | Python | pytest | text formatting, string-heavy diffs |
| sqlparse | BSD-3-Clause | Python | pytest, no dependencies | a parser |
| pyparsing | MIT | Python | unittest | parsers and grammars |
| tomlkit | MIT | Python | pytest | round-trip formatting |
| marshmallow | MIT | Python | pytest | validation and serialization |
| click (repository) | BSD-3-Clause | Python | pytest | approved as a wheel; the repository adds its tests |
| jinja2, markupsafe | BSD-3-Clause | Python | pytest | a template engine |
| rich | MIT | Python | pytest (pygments, markdown-it-py) | larger, terminal rendering |
| attrs | MIT | Python | pytest; skip its Hypothesis tests (Hypothesis is MPL-2.0) | class machinery |
| Django | BSD-3-Clause | Python | its own runner, SQLite | a large real application framework |
| date-fns | MIT | TypeScript | vitest | date arithmetic |
| zod | MIT | TypeScript | vitest | schema validation |
| commander.js | MIT | JavaScript | jest | command-line parsing |
| cobra, testify, gjson | Apache-2.0, MIT, MIT | Go | `go test` | a Go toolchain image needed |
| cJSON | MIT | C | Unity (MIT) | small C library, `make test` |
| regex, serde_json | MIT OR Apache-2.0 | Rust | `cargo test` | a Rust image needed; long builds |
| Elixir at the `v1.18.4` tag | Apache-2.0 | Elixir | ExUnit, `lib/*/test` (thousands of tests, no dependencies) | the library's own tests for the local sources, and Elixir's NOTICE file |
| Jason | Apache-2.0 | Elixir | ExUnit | a JSON parser and encoder |
| Decimal | Apache-2.0 | Elixir | ExUnit | arbitrary-precision arithmetic |
| MIME | Apache-2.0 | Elixir | ExUnit | a small lookup library |
| Earmark Parser | Apache-2.0 | Elixir | ExUnit | a Markdown parser |

Each would be pinned to a release tag and commit, scanned with `license_scan.py`, and entered in `codebases.json`
before any session. Test dependencies count as the environment: any MPL or GPL test dependency (Hypothesis, for
example) keeps those tests out.

## 4. Task generation

| Type | How it is made | Check after the session | Notes |
|---|---|---|---|
| Write tests (stage 1) | per module or package: "write `tests/test_<m>.py` for `<module>`" | module unchanged, the file present, ≥ 5 tests, all pass | gives tests to code bases without them; Flint's tests are publishable |
| Fix a bug (stage 2) | one token changed by `tasks.py`: comparison, arithmetic or boolean operator, a small integer, `True`/`False`, a dropped `not`; kept if 1 to 30 % of the tests fail | test file unchanged, all tests pass | the statement names the failing tests, never the change |
| Implement (stage 2) | a documented public function's body replaced by `raise NotImplementedError`, docstring kept; kept if tests fail | test file unchanged, all tests pass | docstring-to-implementation, as an agent does it |
| Make a failing test pass (next) | a test from the suite whose subject is removed or changed | the named test passes, the rest still pass | from code bases with their own tests |
| Feature (next) | a function and its tests removed; the statement asks for the function as its docstring describes | the hidden tests pass | the hidden tests are the code base's own |
| Refactor (next) | "split", "rename", "extract", "add type hints" on a module | all tests pass, the diff is not empty, `py_compile` | no right answer, only no breakage |
| Explain or investigate (next) | "where is X handled", "why does test Y need Z" | none; or the answer names the right file | read-heavy sessions, realistic and cheap |

**Statement sources.** Templates written by Flint (`texts_spec.json` asks for each kind in six tones and three lengths;
each must contain its placeholders exactly once), filled with names taken from the workspace. Flint can also write a
per-task statement (a user-style bug report from the failing output), checked the same way; it must not contain the
changed line or the fix. Hand-written templates by the owner are the alternative.

**Whose words.** The first texts were written by Flint from requests in prose, and they kept much of that prose: the
three system prompts share a 32-word run with their request, and tool and parameter descriptions took 59–100 % of
their words from theirs. The requests were written by an AI assistant, so the texts carried its wording into every
session. The requests now carry facts as data instead (names, parameters, limits, placeholders, values of at most
four words) under a short instruction, and a text passes only if it shares no run of 5 words with any text the
pipeline's authors wrote; Flint judges each text against its facts (replacing the word-overlap meaning check), and
rewords a text that failed only by copying. The same holds for the tool runner's own lines ("wrote 12 lines to …",
the error messages): Flint writes them, `tools.py` and the agent loop use them, and the sessions gate checks that a
session used texts.json's. The pipeline's own words left in a session are identifiers (tool and parameter names, file
names) and the `error:` prefix of a failed call.

**Risks of Flint-written statements and how they are contained:**
- *Sameness.* One writer converges on few phrasings (the pilot's templates already repeat "without changing the
  tests"). More variants, tones and lengths, a cap per template, and deduplication by 12-word spans.
- *Leakage.* A statement that describes the fix makes the session trivial. The pipeline never shows Flint the change
  when it writes templates; per-task statements are rejected if they contain the changed line.
- *Drift from the request.* Flint may ignore the request: in the pilot's first text run 3 of 13 parameter
  descriptions described a different parameter ("Maximum number of retry attempts…" for a line number). Flint's
  judge pass (the facts and the text: does the text contradict them or leave out a required fact) catches this; a
  reviewer pass before the long run is cheaper than a bad schema in 4,000 sessions.
- *Self-reference.* The model writes the tasks it then solves, so the task distribution follows its own idea of a
  task. Mechanical tasks (mutations, removed bodies, failing tests) are independent of Flint; Flint only phrases them.

## 5. The session generator

- **Our own loop** ([agent.py](../../../bench/agentic_data/agent.py)) with two tool profiles, chosen per session
  (`--tools agx|pi|mixed`): ours (`list_dir`, `read_file`, `search`, `edit_file`, `write_file`, `run_tests`,
  `run_command`) and the pi agent's names and parameters (`read`, `write`, `edit` with a list of replacements,
  `bash`), so the data holds the tool calls of the agent the rig serves. Descriptions come from `texts.json` in both.
  Flint's chat template is rendered here (jinja2, as transformers renders it; compared with llama-server's
  `/apply-template` at every session's start) and tokenized by the server. `--repeat` runs a task again with another
  seed and system prompt.
- **Exact token ids, append-only.** Each turn appends the sampled ids and the tool results as the template renders
  them (`<|observation|><tool_response>…</tool_response>…<|assistant|>`). A tool-calling turn ends with the stop token
  `<|observation|>`, which is also the first token of the results, so the session is one sequence; the reasoning stays
  in the history, as Flint's makers ask. The generated spans are recorded.
- **Sandbox** per session: `--network none`, read-only root, `/work` the only writable folder, a private `/tmp`,
  2 GB, 512 processes, `--cpu-shares 256`. Paths outside `/work` are refused; outputs cut at 20,000 characters.
- **Engines:** llama-server (`/completion` with token ids, `id_slot` per worker so each slot keeps its session's
  cache) or vLLM (`/v1/completions` with token ids; prefix caching must be switched on for multi-turn work, which the
  current serve script turns off).
- **From the agent check:** the Docker sandbox pattern and the task validation idea (the change fails the check,
  the original passes it). Not pi, not its prompts.
- **One format, three consumers** ([export.py](../../../bench/agentic_data/export.py)):

| Consumer | Export | How it reads it |
|---|---|---|
| Answer store (Q161) | `train/turns.jsonl`: one row per turn in `generate.py`'s layout (the exact context as `prompt_tokens`, the turn's `gen_tokens`) and `turn_prompts.jsonl` | `datastore_corpus.py` takes only generated tokens; `datastore build` |
| Drafter dump (Q20) | `train/sessions_seq.jsonl`: one row per session (first prompt, then the rest of the sequence) and `gen_ranges` | `feats.py seqs` and `llama-staircase-dump` replay each session once; training reads targets only inside `gen_ranges` (a `train.py` change); per-turn rows would replay every prefix again |
| Published dataset | `hf/`: messages, token ids, licences, NOTICE, card | the Hub |

## 6. Volume and throughput

**How much.** The store gained little past ~3 M tokens (1.50 → 1.52 from 3 to 13.8 M, Q161); the drafter's heads were
trained on 14–28 M answer tokens and gained ~+0.03 coupled acceptance per doubling. The target is 30–50 M clean
generated tokens: agentic code sessions first, maths (problems only, once the owner rules on OpenR1) at most 10–15 %,
chat (oasst2 root prompts) 10–20 %. The two single-turn parts reuse `generate.py`; the agentic part comes from this
pipeline.

**How many sessions.** Sessions are long: Flint reasons for thousands of tokens before it acts. In the pilot a
write-tests session generated a median 18.2k tokens (10.4–28.0k), a bug repair 3.9k (1.5–18.1k), an implementation
5.2k and 14.6k. A module with three write-tests attempts, ten repairs and three implementations per passing test file
yields est. 120–160k tokens. The standard library has 73 single-file public modules that qualify
(`workspace.py stdlib --modules all`) plus pure-Python packages (`email`, `http`, `urllib`, `xml.dom`, …, not built
yet), so it gives est. 10–15 M tokens; repeating tasks with other seeds (`--repeat`) adds more of the same kind. The
rest of a 30–50 M set needs the approved fsspec and the proposed packages with their own tests (section 8.1), and the
single-turn chat and maths parts.

**How fast** (generated tokens; the pilot's figure is read from its own run log, not a benchmark):

| Writer | Cards | Sessions at once | Generated tok/s | Basis |
|---|---|---|---|---|
| llama.cpp, Flint Q8_0 | two V100s (3, 4) | 8 | ~0.1k (96; 14–16 per session) | the pilot's run log: 149,413 tokens in 25.9 min |
| llama.cpp, Flint Q8_0 | two V100s | 16 | est. 0.15–0.2k | more slots at the same per-step cost |
| vLLM, NVFP4, pipeline-parallel | two RTX 3090s (5, 6) | est. 10–40 | est. 0.3–1.0k | single-turn answers ran 2.4–2.7k tok/s at 160 in flight with ~1.2k-token contexts; vLLM reported 204k tokens of KV cache, so 16–32k-token sessions fit ~6–12 at once if that figure is the real capacity (it is not known whether vLLM frees the 35 sliding-window layers' blocks beyond the window), up to ~40 if it does |
| vLLM, NVFP4, pipeline-parallel | 3080, 3080, 3090 (0, 1, 2) | est. 8–25 | est. 0.2–0.6k | ~1.7 cards' worth of the pair above ([drafter data generation](drafter_data_generation.md)); untested |

- With the RTX 3090s and both vLLM copies: est. 0.6–1.8k tok/s, 2–6.5 M tokens an hour, **30 M tokens in ~5–15
  hours**. With the V100s alone: est. 0.15–0.2k tok/s, 30 M tokens in ~40–55 hours.
- **The binding limit is the KV cache, not the weights:** a session holds 16–64k tokens of context for minutes, so
  far fewer sessions fit than single-turn answers. Prefix caching must be on (the vLLM serve script turns it off);
  without it every turn reads its whole context again.
- **Tools are cheap:** the pilot's 310 tool calls took 0.15 s each on average (container exec included), 48 s for 22
  sessions; 12 CPU cores at a low share are enough for 50+ sessions at once.
- **The dump** replays each session once. In the pilot the sequences were 1.59× the generated tokens (372k for 235k:
  first prompts, tool results). 30 M generated tokens are then ~48 M replayed; at ~1.0–1.65k tok/s on a set whose last
  card sits on a wide link, ~8–13 hours, and ~1.7–2.8 TB at 58 KB per stored position (the 26-layer grid) on the
  bulk disk (3.3 TB free). Today the dump stores every position after the first prompt, tool results included; a
  `DUMP_RANGES` option (store only the generated spans) would cut storage to the generated 1.7 TB.

| GPUs | Generation phase | Dump phase |
|---|---|---|
| 5, 6 (RTX 3090, x8) | vLLM writer | Q8_0 dump on 2, 5, 6 (last card on x8) |
| 0, 1 (RTX 3080) | idle, or with 2 a second vLLM writer (untested) | idle |
| 3, 4 (V100) | llama.cpp Q8_0 writer (the pilot's setup) | a slow Q8_0 dump copy (x1 links, ~0.28k tok/s) |

## 7. Pilot

On 2026-10-26 on the quant machine, 72 minutes on the two V100s (the RTX 3090s were in use by another service):
llama-server with Flint Q8_0 (the approved writer `flint-q8_0`; 16 slots of 32,768 tokens, q8_0 KV cache), the
sandbox image, and eight standard-library modules (graphlib, getopt, reprlib, colorsys, fnmatch, shlex, fractions,
netrc). Everything ran through `run.sh` and the gates; the sessions are a technical check, not training data, until
the owner approves the texts.

| Step | Result |
|---|---|
| Scan, workspaces | 8 modules with a plain `ok` verdict; workspaces gate passed |
| Texts | Flint wrote 37 texts for our tool set and 17 for the pi tool set; every one passed its checks on the first try (after the fix in item 3 below) |
| Stage 1: write tests | 8 sessions, 135 turns, 149,413 generated tokens; 4 verified (colorsys, getopt, graphlib, netrc: 50, 46, 34 and 18 tests, all passing on the unchanged module); ends: 3 answers, 2 turn caps, 2 context limits, 1 time cap |
| Stage 2: repairs and implementations | from the 4 test files: 12 single-token bugs (an operator, an index, a constant) and 2 removed function bodies; 14 sessions, 85,256 generated tokens, **14 of 14 verified** (repairs: median 3.9k tokens, 7.5 turns, 5 minutes) |
| Gates | workspaces, three task sets, 22 sessions, the store and dump exports (284 rows), the turn prompts (262), the dataset (22 rows): all passed; the corpus and the store recorded through `check_sources.py derived` |
| Store | `datastore_corpus.py` and `datastore build` on the turns: 234,931 tokens, a `PIPESTO1` file |
| Dump | `feats.py seqs` read the 22 session sequences (372k tokens for 235k generated); the dump itself was not run |
| Token chain | 243 of 243 tool-calling turns ended with `<|observation|>`, the first token of the results, so every session is one exact sequence; the template rendered as llama-server's in 22 of 22 sessions |

What broke, and what changed:
1. **8,192 tokens per turn were too few.** Flint reasoned for ~29,000 characters about getopt before writing a line of
   the test file, and the turn was cut. Raised to 20,000; two sessions then ran into the 32,768-token context. The long
   run needs 64k contexts and ≥ 32k tokens per turn (the makers ask for 32,768 or more on hard problems).
2. **Our tool names cost calls.** With our tool set 18 of 173 calls (10 %) went to tools that do not exist (`shell`
   16 times, `terminal`, `shell_command`); Flint read the error and switched to `run_command`. It never used
   `run_tests` or `search`. With the pi agent's names (`read`, `write`, `edit`, `bash`) none of 137 calls missed
   (stage 2 only, so task type is mixed in). The pi tool set now exists as a profile: the store and the drafter then
   learn the tool calls of the agent the rig is measured with (the store's tool-call gain came only from answers that
   used the same tools, Q161).
3. **Flint-written texts can drift:** in the first text run 3 of 13 parameter descriptions described other
   parameters ("Maximum number of retry attempts…" for a line number). Each text must now share half of its
   request's content words; the tool and parameter names go into the request.
4. **Templates echo their instructions:** "Do not guess what the bug is", meant for the template writer, came back as
   "don't guess" in users' voices. The request now says the statement must not hint at the bug; the texts get the
   owner's review either way.
5. **Gate false alarms, all on provenance of tool output:** scratch files in `/tmp`, the interpreter's own message
   ("/usr/local/bin/python: No module named pytest") and the shell's ("/bin/sh: 1: pytest: not found") excluded
   sessions. These are allowed now; anything else outside the project still excludes a session. The tasks gate
   also refused a placeholder for the file a task creates (fixed).
6. **Flint expects pytest.** It tried `python -m pytest` in several sessions; the environment only has unittest. The
   `py312-pytest` environment (section 8.1) matches what agents meet in practice.
7. **Implementation tasks were filtered out** by the cap meant for one-token bugs (at most 30 % of tests failing); a
   removed body may fail them all. Fixed.
8. **`check_sources.py` could not record the store**: its `derived` stage took the compiled `datastore` program for
   untraced data. It now takes `--tool` (programs built from this repository, recorded by hash).
9. **A model failure worth keeping:** in reprlib Flint wrote `Rrepr` for `Repr` a dozen times, noticed it ("my
   generated text keeps producing Rrepr"), and ran out of turns. Such sessions stay in the data, marked unverified.

Not covered: the vLLM writer (the RTX 3090s were busy), prefix caching in vLLM, the dump itself, code bases other
than the standard library, the agent loop against a 64k context.

## 8. The long run

### 8.1 Approvals first (the owner)

| # | Decision | Why | Default if not given |
|---|---|---|---|
| 1 | The harness texts, written by Flint from texts_spec.json's facts (version 2: 5 system prompts, 13 tool and 26 parameter descriptions, 28 statement templates and the change note for Python and Elixir, 32 tool runner messages); the first texts.json is retired because it reused the requests' wording | they are in every session | the run does not start |
| 2 | Retire the AI-assistant-written texts of `prompts.py` and `clean_prompts.py` (`q20-task-texts` in `sources.json`) for all new data; record text writers in `sources.json` | Anthropic's usage policy and the owner's own rule | they stay unused |
| 3 | Free the RTX 3090s (`./rig apply off`, which stops the `flint-q8-262k` preset) for the run's hours | the preset held GPUs 2, 5, 6 and was in use | the run uses the V100s only (~40–55 h for 30 M tokens) |
| 4 | The `py312-pytest` environment (pytest, pluggy, iniconfig, pygments, packaging: MIT, BSD; copied from the agent check's image, no download) | fsspec's tests and most packages' tests need pytest | approved 2026-10-26 |
| 5 | Test folders of the approved trees (llama.cpp `tests/`, `gguf-py/tests`; fsspec's tests), same licence as the trees | `sources.json` excludes test folders, a filter for the old prompt files | approved 2026-10-26 |
| 6 | The local packages: SymPy 1.14.0, NetworkX 3.7, mpmath 1.3.0, jsonschema 4.26.0 (+ referencing), dill 0.4.1, NumPy 2.2.6 (in the vLLM image, scanned) | variety beyond the standard library, real test suites | approved 2026-10-26, test folders included |
| 6a | Elixir: the library sources of the local `elixir:1.18.4-otp-27` image (Apache-2.0, 246 files `ok`), the `ex118-sandbox` environment built from it, and Elixir's NOTICE file from the `v1.18.4` tag (one text file); later OTP's sources, and the Elixir downloads of section 3.2 | a second language, the one many of the rig's users write | approved 2026-10-26 (the library, the sandbox, the NOTICE); OTP and the downloads not yet |
| 7 | Downloads from section 3.2 (each pinned and scanned before use) | languages other than Python, larger projects | not used |
| 8 | OpenR1-Math: keep only rows whose `source` is not synthetic, and accept that contest problem texts come from forums and exam papers, or drop maths | the approved split mixes in synthetic problems | maths stays out |
| 9 | Hugging Face: dataset name, licence metadata (`apache-2.0`, `other`/python-2.0, plus each code base's), whether token ids are published | publication | nothing is uploaded |
| 10 | (optional) ScanCode Toolkit as a second scanner (a pip install in a container) | a stronger check than the signature scan | the signature scan only |

Technical work before the run, no approval needed:
- a prefix-cache switch in `vllm/serve_vllm.sh` (it passes `--no-enable-prefix-caching`) and 64k-token contexts
  (`MAX_LEN=65536`);
- a short vLLM trial of the agent loop (~10 sessions): it has only run against llama-server; check the prefix cache
  with the patched Flint attention (the same turns with and without it, greedy) and read the KV-cache use from vLLM's
  log, which decides how many sessions fit;
- `train.py` reading only `gen_ranges` (and optionally `DUMP_RANGES` in the dump);
- for more than ~15 M tokens: standard-library packages in `workspace.py`, and tasks on code bases with their own
  test suites (`tasks.py` today derives stage 2 from Flint-written tests only);
- Elixir (after 6a): `workspace.py elixir` (one module per Mix project, the rename and the changed-file line),
  mutations and removed bodies through an Elixir script in the sandbox, a doctest check in the stage-2 verification,
  Elixir variants of the write-tests texts in `texts_spec.json` (ExUnit, `mix test`), then a pilot of ~10 sessions to
  see whether Flint writes passing ExUnit tests before Elixir joins the long run.

### 8.2 The run

From a copy of the repository's `bench/` folder on the quant machine, with the machine's paths in the environment:

```bash
# 1. the vLLM writer on two RTX 3090s, 64k contexts, prefix caching on (after the switch above)
GPUS=5,6 MODEL_DIR=<flint-nvfp4 folder> HF_DIR=<RecursiveSI/flint small files> WORK=<vllm work folder> \
  MAX_LEN=65536 MAX_SEQS=48 bash bench/staircase_drafter/vllm/serve_vllm.sh
# 2. the pipeline against it: every eligible standard-library module, three stage-1 attempts, up to ten repairs and
#    three implementations per passing test file, both tool profiles, the reviewed texts
WORK=<bulk disk>/agentic-clean/run1 API=vllm PORT=18560 WRITER=flint-nvfp4-vllm MODEL=<.../flint-nvfp4.gguf> \
  HF_DIR=<...> Q51=<Q51 prompts.jsonl> TEXTS=bench/agentic_data/texts.json MODULES=all ATTEMPTS=3 PER_MODULE=10 \
  IMPLEMENT=3 TOOLS=mixed CONC=32 CTX_SLOT=65536 MAX_GEN=32768 TIME_CAP=45 CLAIMS=<bulk disk>/agentic-clean/claims \
  STEPS="image workspaces s1 s2 gate export checks" bash bench/agentic_data/run.sh
# 3. optional second writer on the V100s, the same task list: the shared CLAIMS folder gives each task to one writer
WORK=<bulk disk>/agentic-clean/run1b GPUS=3,4 NP=8 CTX_SLOT=65536 MODEL=<flint-Q8_0.gguf> LLAMA_SERVER=<llama-server> \
  MOUNTS="..." HF_DIR=<...> Q51=<...> TEXTS=bench/agentic_data/texts.json MODULES=all ATTEMPTS=3 PER_MODULE=10 \
  IMPLEMENT=3 MAX_GEN=32768 CLAIMS=<bulk disk>/agentic-clean/claims \
  STEPS="image workspaces server s1 s2 gate export checks stop" bash bench/agentic_data/run.sh
```

With one writer the defaults do everything. With two, each run gates and exports its own sessions, and a last
`gate.py sessions` over both runs' session folders (all four task records as parents) and one `export.py` give the
merged set. Stage 2 starts when a run's own stage 1 has finished, from its own sessions.

**Monitoring:**
- `tail -f <WORK>/s1/run.log <WORK>/s2/run.log`: one line per session (turns, generated tokens, end, verified) and
  running totals;
- `python3 bench/agentic_data/report.py <WORK>/s1/sessions <WORK>/s2/sessions`: totals, ends, verified share, calls
  by tool, failed calls;
- vLLM's log (`docker logs pq-vllm-gen`): generation throughput, KV-cache use, preemptions;
- `df -h` on the fast and the bulk disk, and `nvidia-smi`.

**Stop conditions:**
- graceful: `touch <WORK>/STOP` (no new sessions; running ones finish; the gate and the exports then run);
- stop and look when more than 5 % of sessions end as `.error.json` (infrastructure), the gate fails (a rule broke:
  stop at once), more than 10 % are excluded by the content checks, vLLM preempts, the NVMe falls under 50 GB, or the
  owner needs the machine;
- done at 30–35 M generated tokens in passed sessions, or when the task list is used up.

**The gate at the end:**
1. `gate.py sessions` over every session folder → `export.py` → `gate.py answers` and `gate.py dataset`;
2. the owner reads ~20 random sessions of `hf/data/sessions.jsonl`;
3. the store: `datastore_corpus.py` + `datastore build` on `train/turns.jsonl`, recorded as `run.sh`'s checks step
   does (`check_sources.py derived`: the corpus from the answers and turn-prompt records, then the store with
   `--tool` for the datastore program);
4. the dump: `feats.py seqs` on `train/sessions_seq.jsonl`, `llama-staircase-dump` with Q8_0, then
   `check_sources.py dump --parent answers.gate.json`; heads train on the generated spans only.

## Sources

Licences and policies (primary):
- MIT: [opensource.org/license/mit](https://opensource.org/license/mit); BSD-2-Clause and BSD-3-Clause:
  [opensource.org/license/bsd-2-clause](https://opensource.org/license/bsd-2-clause),
  [opensource.org/license/bsd-3-clause](https://opensource.org/license/bsd-3-clause)
- Apache License 2.0, §4 and the appendix: [apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)
- PSF License Agreement and the history of Python's licence: [docs.python.org/3/license.html](https://docs.python.org/3/license.html);
  CPython 3.12.15's `LICENSE.txt` (in the image)
- ASF third-party licence policy: [apache.org/legal/resolved.html](https://www.apache.org/legal/resolved.html);
  [infra.apache.org/licensing-howto.html](https://infra.apache.org/licensing-howto.html)
- SPDX 2.3: [short identifiers in source files](https://spdx.github.io/spdx-spec/v2.3.1/using-SPDX-short-identifiers-in-source-files/),
  [licence expressions](https://spdx.github.io/spdx-spec/v2.3.1/SPDX-license-expressions/),
  [Python-2.0](https://spdx.org/licenses/Python-2.0.html), [PSF-2.0](https://spdx.org/licenses/PSF-2.0.html);
  REUSE 3.3: [reuse.software/spec-3.3](https://reuse.software/spec-3.3/)
- GNU GPL FAQ on output: [gnu.org/licenses/gpl-faq.html#GPLOutput](https://www.gnu.org/licenses/gpl-faq.html#GPLOutput)
  (could not be fetched on 2026-10-26; cited from its long-standing wording)
- US Copyright Office: [88 FR 16190](https://www.govinfo.gov/content/pkg/FR-2023-03-16/html/2023-05321.htm) (2023);
  [Copyright and Artificial Intelligence, Part 2](https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-2-Copyrightability-Report.pdf) (2025)
- Anthropic: [usage policy](https://www.anthropic.com/legal/aup), [commercial terms](https://www.anthropic.com/legal/commercial-terms),
  [consumer terms](https://www.anthropic.com/legal/consumer-terms); OpenAI terms of use (2024-12-11; openai.com refused
  the fetch, read from a mirror); Google [Gemini API additional terms](https://ai.google.dev/gemini-api/terms)
- GitHub terms of service, section D: [docs.github.com](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service)
- Hugging Face: [dataset cards](https://huggingface.co/docs/hub/datasets-cards),
  [licences](https://huggingface.co/docs/hub/repositories-licenses)
- Data sets: [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2),
  [open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k),
  [AI-MO/NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5)
- Flint: [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint) (model card, `LICENSE`, `chat_template.jinja`,
  `tokenizer_config.json`, `generation_config.json`)

This repository: [drafter](../../drafter.md), [drafter data generation](drafter_data_generation.md),
[agent prompt cache](agent_prompt_cache.md), [bench/staircase_drafter](../../../bench/staircase_drafter/README.md),
[bench/ngram_replay](../../../bench/ngram_replay/README.md), [bench/agent_check](../../../bench/agent_check/README.md),
registry rows Q20, Q51 and Q161 in [quants/experiments.md](../../../quants/experiments.md).
