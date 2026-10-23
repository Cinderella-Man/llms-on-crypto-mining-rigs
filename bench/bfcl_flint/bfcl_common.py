"""Shared pieces: BFCL v4 imports (from the bfcl-eval 2026.3.23 wheel, unpacked) and Flint's tool-call parser.
Template rendering and the llama-server client with the makers' sampling are in ../flint_client.py.

The parser follows the one in Recursive's own engine (rec, engine/src/rec_mlx/chat.py, parse_tool_call): a call is
`<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>`, a value stays a raw string when the
tool's schema accepts a string and is parsed as JSON otherwise (kept raw if that fails), and `null` becomes None only
when the schema accepts null. Reasoning is the `<think>` block; calls inside it are not calls.

Heavy modules that the checker never uses for Python categories are stubbed:
- bfcl_eval.constants.model_config (imports every API SDK): replaced by a mapping whose entries all say
  underscore_to_dot=True, the setting BFCL uses for OpenAI-style FC models (tool names have "." replaced by "_").
- the Java/JavaScript tree-sitter parsers.
"""
import copy
import json
import os
import re
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flint_client  # noqa: E402
from flint_client import (SAMPLING, complete, count_tokens, make_template, parse_attempts,  # noqa: E402,F401
                          server_props, sha256_of)

BFCL_ROOT = os.environ.get("BFCL_ROOT", "/bfcl/pkgs/v4")  # the unpacked bfcl-eval==2026.3.23 wheel
sys.path.insert(0, BFCL_ROOT)


class _Cfg:
    underscore_to_dot = True


class _Map(dict):
    def __missing__(self, key):
        return _Cfg()


_m = types.ModuleType("bfcl_eval.constants.model_config")
_m.MODEL_CONFIG_MAPPING = _Map()
sys.modules["bfcl_eval.constants.model_config"] = _m
for _name, _fn in [
    ("bfcl_eval.model_handler.parser.java_parser", "parse_java_function_call"),
    ("bfcl_eval.model_handler.parser.js_parser", "parse_javascript_function_call"),
]:
    _mm = types.ModuleType(_name)
    setattr(_mm, _fn, None)
    sys.modules[_name] = _mm

from bfcl_eval.constants.default_prompts import (  # noqa: E402
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
    MAXIMUM_STEP_LIMIT,
)
from bfcl_eval.constants.enums import Language, ModelStyle  # noqa: E402
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI  # noqa: E402
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker  # noqa: E402
from bfcl_eval.model_handler.utils import convert_to_function_call, convert_to_tool  # noqa: E402
from bfcl_eval.utils import (  # noqa: E402
    is_empty_output,
    is_function_calling_format_output,
    load_dataset_entry,
    load_ground_truth_entry,
)

BFCL_VERSION = "bfcl-eval 2026.3.23"
MODEL = "flint-FC"

# BFCL v4's judge-free single-turn Python categories (3,491 tasks) and its multi-turn categories (800 tasks)
SINGLE_TURN = [
    "simple_python",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "live_relevance",
]
MULTI_TURN = ["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"]


def live_cluster(task_id):
    """Live tasks are named <category>_<index>-<source>-<n>; tasks from one source share a cluster."""
    m = re.match(r"(live_\w+?)_\d+-(\d+)-\d+$", task_id)
    return f"{m.group(1)}-{m.group(2)}" if m else task_id


# ---------------------------------------------------------------- tools and parser
def build_tools(functions):
    """BFCL's tool list for an OpenAI chat-completions client (the language hint is already in the docs)."""
    return convert_to_tool(copy.deepcopy(functions), GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)


_CALL = re.compile(r"\s*([^<\s]+)\s*((?:<arg_key>.*?</arg_key>\s*<arg_value>.*?</arg_value>\s*)*)", re.S)
_ARGUMENT = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)
_MARKERS = ("<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", "<think>",
            "</think>", "<|")


def _accepts(schema, kind):
    types_ = schema.get("type")
    if types_ == kind or (isinstance(types_, list) and kind in types_):
        return True
    if kind == "string" and any(isinstance(value, str) for value in schema.get("enum", ())):
        return True
    return any(_accepts(option, kind) for key in ("anyOf", "oneOf") for option in schema.get(key, ()))


def _tool_properties(tools):
    out = {}
    for t in tools or ():
        f = t.get("function", t)
        out[f["name"]] = f.get("parameters", {}).get("properties", {})
    return out


def parse_tool_call(text, properties_by_name):
    """rec's parse_tool_call: the text between <tool_call> and </tool_call>, typed by the tool's schema."""
    call = _CALL.fullmatch(text)
    if call is None:
        raise ValueError(f"Malformed Flint tool call: {text!r}")
    name = call[1]
    properties = properties_by_name.get(name, {})
    arguments = {}
    for key, value in _ARGUMENT.findall(call[2]):
        schema = properties.get(key, {})
        if value == "null" and _accepts(schema, "null"):
            value = None
        elif not _accepts(schema, "string"):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        arguments[key] = value
    return {"name": name, "arguments": arguments}


def split_output(text, tools):
    """Split one generated assistant turn into reasoning, content and tool calls, and flag process failures.

    The reasoning runs to the last </think>. rec's grammar allows a single </think>; without it Flint sometimes
    closes its reasoning, thinks on and closes it again, and llama-server still finds the calls after it.

    Returns a dict: reasoning, content, calls ([{name, arguments}]), and the flags
    - malformed: a call that rec's parser rejects, a <tool_call> without </tool_call>, or a stray protocol marker
      outside the reasoning (the decoder then fails, as BFCL counts an undecodable answer);
    - think_open: the reasoning never closed (the turn ran out of tokens while thinking);
    - extra_think_close: </think> written more than once;
    - call_in_think: a <tool_call> inside the reasoning (it is not a call);
    - unknown_function / unknown_arg: number of calls to a function that is not offered, and of argument names
      that are not in the function's schema."""
    reasoning, content = "", text
    think_open = False
    if "</think>" in text:
        reasoning, content = text.rsplit("</think>", 1)
        reasoning = reasoning.split("<think>", 1)[-1]
    elif text.lstrip().startswith("<think>"):
        reasoning, content, think_open = text.lstrip()[len("<think>"):], "", True
    props = _tool_properties(tools)
    calls, outside, malformed = [], [], False
    rest = content
    while "<tool_call>" in rest:
        before, rest = rest.split("<tool_call>", 1)
        outside.append(before)
        body, sep, rest = rest.partition("</tool_call>")
        if not sep:
            malformed = True
            break
        try:
            calls.append(parse_tool_call(body, props))
        except ValueError:
            malformed = True
    outside.append(rest)
    plain = "".join(outside)
    if any(m in plain for m in _MARKERS):
        malformed = True
    return {
        "reasoning": reasoning,
        "content": content if malformed else plain,
        "calls": calls,
        "malformed": malformed,
        "think_open": think_open,
        "extra_think_close": "</think>" in reasoning,
        "call_in_think": "<tool_call>" in reasoning,
        "unknown_function": sum(c["name"] not in props for c in calls),
        "unknown_arg": sum(k not in props.get(c["name"], {}) for c in calls if c["name"] in props
                           for k in c["arguments"]),
    }


def decode_ast(parsed):
    """BFCL's AST format [{name: {arg: value}}]; an undecodable answer raises, as a handler's decode_ast does."""
    if parsed["malformed"]:
        raise ValueError("malformed tool call")
    return [{c["name"]: c["arguments"]} for c in parsed["calls"]]


def assistant_message(parsed):
    """The assistant turn as an OpenAI-style message, the way a client of rec's API sends it back: reasoning kept."""
    msg = {"role": "assistant", "content": parsed["content"]}
    if parsed["reasoning"]:
        msg["reasoning_content"] = parsed["reasoning"]
    if parsed["calls"] and not parsed["malformed"]:
        msg["tool_calls"] = [{"type": "function", "function": c} for c in parsed["calls"]]
    return msg


def manifest(url, extra):
    return flint_client.manifest(url, {"bfcl": BFCL_VERSION, **extra})
