"""Shared pieces: BFCL v3 checker imports (from the bfcl-eval 3.0.0 wheel, unpacked),
Flint chat-template rendering (same Jinja environment as HF transformers) and the
Flint tool-call decoder.

Heavy modules that the checker never uses for Python categories are stubbed:
- bfcl.constants.model_config (imports every API SDK): replaced by a mapping whose
  entries all say underscore_to_dot=True, the setting BFCL uses for OpenAI-style FC
  models (tool names have "." replaced by "_").
- the Java/JavaScript tree-sitter parsers.
"""
import os
import ast
import copy
import json
import re
import sys
import types

BFCL_ROOT = os.environ.get("BFCL_ROOT", "/bfcl/pkgs/v3")  # the unpacked bfcl-eval==3.0.0 wheel
sys.path.insert(0, BFCL_ROOT)


class _Cfg:
    underscore_to_dot = True


class _Map(dict):
    def __missing__(self, key):
        return _Cfg()


_m = types.ModuleType("bfcl.constants.model_config")
_m.MODEL_CONFIG_MAPPING = _Map()
sys.modules["bfcl.constants.model_config"] = _m
for _name, _fn in [
    ("bfcl.model_handler.parser.java_parser", "parse_java_function_call"),
    ("bfcl.model_handler.parser.js_parser", "parse_javascript_function_call"),
]:
    _mm = types.ModuleType(_name)
    setattr(_mm, _fn, None)
    sys.modules[_name] = _mm

from bfcl.eval_checker.ast_eval.ast_checker import ast_checker  # noqa: E402
from bfcl.model_handler.utils import (  # noqa: E402
    convert_to_tool,
    func_doc_language_specific_pre_processing,
)
from bfcl.constants.type_mappings import GORILLA_TO_OPENAPI  # noqa: E402
from bfcl.model_handler.model_style import ModelStyle  # noqa: E402
from bfcl.utils import is_function_calling_format_output, is_empty_output  # noqa: E402

CATEGORIES = [
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "live_simple",
    "live_multiple",
    "irrelevance",
    "live_irrelevance",
]
# Per-category counts inferred from terelabs' comparison.csv (their percentages are
# multiples of 2, 2.5, 2.5, 2.5, 2.5, 2.5, 5, 5 and the counts must add up to 300).
COUNTS = {
    "simple": 50,
    "multiple": 50,
    "parallel": 40,
    "parallel_multiple": 40,
    "live_simple": 40,
    "live_multiple": 40,
    "irrelevance": 20,
    "live_irrelevance": 20,
}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- chat template
def make_template(template_str):
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    def raise_exception(message):
        raise RuntimeError(message)

    # Same settings as transformers' _compile_jinja_template
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env.from_string(template_str)


def build_tools(functions, category):
    """BFCL FC-mode tool list for an OpenAI-compatible client."""
    fn = func_doc_language_specific_pre_processing(copy.deepcopy(functions), category)
    return convert_to_tool(fn, GORILLA_TO_OPENAPI, ModelStyle.OpenAI)


# ---------------------------------------------------------------- decoder
_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)
_ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)


def _convert(value, ptype):
    v = value.strip()
    if v == "null":  # the template writes None as null, whatever the declared type
        return None
    if ptype == "string":
        return v
    try:
        return json.loads(v)
    except Exception:
        pass
    try:
        return ast.literal_eval(v)
    except Exception:
        return v


def decode_flint(text, tools):
    """Flint / GLM-4.5 style: <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>.
    The template writes string values raw and everything else as JSON, so values whose
    schema type is string are kept verbatim (except a bare null, which becomes None) and
    the rest are parsed as JSON (then as a Python literal, then kept as a string)."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    schema = {}
    for t in tools:
        f = t.get("function", t)
        schema[f["name"]] = f.get("parameters", {}).get("properties", {})
    calls = []
    for m in _CALL_RE.finditer(text):
        body = m.group(1)
        i = body.find("<arg_key>")
        name = (body if i < 0 else body[:i]).strip()
        props = schema.get(name, {})
        args = {}
        for km in _ARG_RE.finditer(body):
            k = km.group(1).strip()
            args[k] = _convert(km.group(2), props.get(k, {}).get("type"))
        calls.append({name: args})
    return calls
