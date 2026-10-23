"""Shared pieces of the LiveCodeBench harness: problems from code_generation_lite (fetch.py), and LiveCodeBench's own
prompt, code extraction and test runner, imported unmodified from a LiveCodeBench checkout (LCB_ROOT, commit
28fef95). Flint's template and the llama-server client with the makers' sampling are in ../flint_client.py.
"""
import base64
import io
import json
import os
import pickle
import sys
import types
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flint_client  # noqa: E402
from flint_client import (SAMPLING, complete, count_tokens, make_template, parse_attempts,  # noqa: E402,F401
                          server_props, sha256_of)

LCB_ROOT = os.environ.get("LCB_ROOT", "/lcb/LiveCodeBench")  # git checkout of LiveCodeBench/LiveCodeBench
LCB_COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
DATASET = "livecodebench/code_generation_lite release_v6 @ 0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
sys.path.insert(0, LCB_ROOT)

# lcb_runner.benchmarks imports the Hugging Face datasets package only to download (fetch.py does that instead),
# and lcb_runner.prompts imports anthropic's prompt markers, which code generation does not use.
_ds = types.ModuleType("datasets")
_ds.load_dataset = None
sys.modules.setdefault("datasets", _ds)
_an = types.ModuleType("anthropic")
_an.HUMAN_PROMPT = _an.AI_PROMPT = None
sys.modules.setdefault("anthropic", _an)
_cwd = os.getcwd()
os.chdir(LCB_ROOT)  # lcb_runner.prompts reads its few-shot files by relative path at import time
try:
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem  # noqa: E402
    from lcb_runner.evaluation.compute_code_generation_metrics import evaluate_generations  # noqa: E402,F401
    from lcb_runner.lm_styles import LMStyle  # noqa: E402
    from lcb_runner.prompts.code_generation import format_prompt_generation  # noqa: E402
    from lcb_runner.utils.extraction_utils import extract_code  # noqa: E402
finally:
    os.chdir(_cwd)

# LiveCodeBench's runner style for an OpenAI-compatible chat model: its generic system message, then the question
# with the starter code (LeetCode) or the stdin/stdout instruction (AtCoder, Codeforces).
STYLE = LMStyle.OpenAIChat
DIFFICULTIES = ["easy", "medium", "hard"]


class _NoGlobals(pickle.Unpickler):
    """The hidden tests are a pickled str; refuse anything that would import or call code."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def decode_tests(field):
    try:
        return json.loads(field)
    except ValueError:
        return json.loads(_NoGlobals(io.BytesIO(zlib.decompress(base64.b64decode(field)))).load())


def load_problems(path, ids=None, with_tests=True):
    """CodeGenerationProblem objects from a fetch.py file, one at a time (the tests of 454 problems take GBs)."""
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if ids is not None and row["question_id"] not in ids:
                continue
            row["private_test_cases"] = json.dumps(decode_tests(row["private_test_cases"]) if with_tests else [])
            yield CodeGenerationProblem(**row)


def messages(problem):
    return format_prompt_generation(problem, STYLE)


def final_answer(text):
    """The part of a generated turn after the reasoning, which is what rec's API returns as content: the text after
    the last </think> (a turn that never closed its reasoning has no answer)."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1], False
    if text.lstrip().startswith("<think>"):
        return "", True
    return text, False


def code_of(text):
    """(code, think_open): LiveCodeBench's extraction (the last fenced block) applied to the final answer."""
    answer, think_open = final_answer(text)
    return extract_code(answer, STYLE), think_open


def manifest(url, extra):
    return flint_client.manifest(url, {"livecodebench_commit": LCB_COMMIT, "dataset": DATASET, **extra})
