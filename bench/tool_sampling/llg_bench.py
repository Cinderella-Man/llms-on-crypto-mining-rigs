"""Token-mask cost of llguidance for Flint's tool-call grammar (150,144-token vocabulary).

Usage: python -I llg_bench.py tokenizer.json flint_tools.gbnf
"""
import json
import statistics
import sys
import time

from llguidance import LLMatcher, LLTokenizer
from llguidance.gbnf_to_lark import gbnf_to_lark

tok_json = open(sys.argv[1]).read()
gbnf = open(sys.argv[2]).read()

t0 = time.perf_counter()
tok = LLTokenizer(tok_json, n_vocab=150144, eos_token=[150001, 150003, 150005, 150018])
print(f"tokenizer: {time.perf_counter() - t0:.2f} s, vocab {tok.vocab_size}")

code = open(__file__.replace("llg_bench.py", "stats_code.txt")).read()
call1 = ("<tool_call>write<arg_key>filePath</arg_key><arg_value>src/stats.py</arg_value>"
         "<arg_key>content</arg_key><arg_value>" + code + "</arg_value></tool_call>")
call2 = ('<tool_call>todowrite<arg_key>todos</arg_key><arg_value>[{"content": "Write src/stats.py", "status": "completed", '
         '"priority": "high", "id": "1"}, {"content": "Add tests", "status": "pending", "priority": "medium", "id": "2"}]'
         '</arg_value></tool_call>')

# An idiomatic llguidance grammar for the same two tools: raw string values are lazy lexemes that stop at </arg_value>.
lark_hand = r"""
start: call+
call: <[150010]> (write | todowrite) <[150011]>
write: "write" <[150012]> "filePath" <[150013]> <[150014]> RAW <[150015]> <[150012]> "content" <[150013]> <[150014]> RAW <[150015]>
todowrite: "todowrite" <[150012]> "todos" <[150013]> <[150014]> %json {
  "type": "array",
  "items": {"type": "object",
            "properties": {"content": {"type": "string"},
                           "status": {"enum": ["pending", "in_progress", "completed", "cancelled"]},
                           "priority": {"enum": ["high", "medium", "low"]},
                           "id": {"type": "string"}},
            "required": ["content", "status", "priority", "id"], "additionalProperties": false}
} <[150015]>
RAW: /(?s:.*)/ & ~/(?s:.*)<\/arg_value>(?s:.*)/
"""


def run(name, grammar, text_tokens):
    t0 = time.perf_counter()
    m = LLMatcher(tok, grammar, log_level=1)
    t_build = time.perf_counter() - t0
    if m.is_error():
        print(f"{name}: grammar error: {m.get_error()[:300]}")
        return
    times, consume = [], []
    for t in text_tokens:
        t0 = time.perf_counter()
        mask = m.compute_bitmask()
        times.append((time.perf_counter() - t0) * 1e3)
        allowed = (mask[t // 8] >> (t % 8)) & 1
        if not allowed:
            print(f"{name}: token {t} {tok.decode_str([t])!r} not allowed; stop")
            break
        t0 = time.perf_counter()
        ok = m.consume_token(t)
        consume.append((time.perf_counter() - t0) * 1e3)
        if not ok or m.is_error():
            print(f"{name}: consume failed at {t}: {m.get_error()[:200]}")
            break
    times_s = sorted(times)
    p = lambda q: times_s[min(len(times_s) - 1, int(q * len(times_s)))]
    print(f"{name}: build {t_build * 1e3:.0f} ms, {len(times)} masks: mean {statistics.mean(times):.3f} ms, "
          f"p50 {p(0.5):.3f}, p90 {p(0.9):.3f}, p99 {p(0.99):.3f}, max {max(times):.3f}; "
          f"consume mean {statistics.mean(consume):.4f} ms; mask {len(mask)} bytes")


toks = tok.tokenize_bytes((call1 + call2).encode(), parse_special=True)
print(f"{len(toks)} tokens in the two calls")
run("hand-written lark", lark_hand, toks)

lark_auto = gbnf_to_lark(gbnf)
open(__file__.replace("llg_bench.py", "flint_tools.lark"), "w").write(lark_auto)
run("gbnf_to_lark(llama.cpp grammar)", lark_auto, toks)

# The full nine-tool set of tool_sampling_bench.cpp, generated in the shape llama.cpp's PEG grammar has:
# required arguments in order, then optional ones in any order; strings as lazy lexemes, other types as %json.
TOOLS = {
    "bash": ({"command": "string", "timeout": "number", "description": "string"}, ["command", "description"]),
    "read": ({"filePath": "string", "offset": "number", "limit": "number"}, ["filePath"]),
    "write": ({"filePath": "string", "content": "string"}, ["filePath", "content"]),
    "edit": ({"filePath": "string", "oldString": "string", "newString": "string", "replaceAll": "boolean"},
             ["filePath", "oldString", "newString"]),
    "glob": ({"pattern": "string", "path": "string"}, ["pattern"]),
    "grep": ({"pattern": "string", "path": "string", "include": "string"}, ["pattern"]),
    "list": ({"path": "string", "ignore": {"type": "array", "items": {"type": "string"}}}, []),
    "todowrite": ({"todos": {"type": "array", "items": {"type": "object", "properties": {
        "content": {"type": "string"}, "status": {"enum": ["pending", "in_progress", "completed", "cancelled"]},
        "priority": {"enum": ["high", "medium", "low"]}, "id": {"type": "string"}},
        "required": ["content", "status", "priority", "id"], "additionalProperties": False}}}, ["todos"]),
    "webfetch": ({"url": "string", "format": {"enum": ["text", "markdown", "html"]}, "timeout": "number"}, ["url", "format"]),
}
lines = ["start: call+", "call: <[150010]> WS? (" + " | ".join(f"t_{n}" for n in TOOLS) + ") WS? <[150011]> WS?",
         r"WS: /[ \n\t]+/", r"RAW: /(?s:.*)/ & ~/(?s:.*)<\/arg_value>(?s:.*)/"]
for name, (props, req) in TOOLS.items():
    arg_rules = []
    for k, ty in props.items():
        r = f"a_{name}_{k.lower()}"
        if ty == "string":
            val = "RAW"
        else:
            schema = {"type": ty} if isinstance(ty, str) else ty
            val = "%json " + json.dumps(schema)
        lines.append(f'{r}: <[150012]> "{k}" <[150013]> <[150014]> {val} <[150015]>')
        arg_rules.append(r)
    seq = " WS? ".join(f"a_{name}_{k.lower()}" for k in req)
    opt = [f"a_{name}_{k.lower()}" for k in props if k not in req]
    body = f'"{name}" WS? ' + seq + (f" (WS? ({' | '.join(opt)}))*" if opt else "")
    lines.append(f"t_{name}: {body}")
lark_full = "\n".join(lines)
run("generated lark, nine tools", lark_full, toks)
