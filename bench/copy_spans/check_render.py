"""Optional check of render_prompt(): render every request with the model's own Jinja chat template and compare the
text byte for byte (needs the jinja2 package; the analysis itself does not).

    RUNS="baseline=/path/run1 ..." TEMPLATE=/path/chat_template.jinja python3 check_render.py

TEMPLATE is the chat_template.jinja of the model repository (or the template stored in the GGUF). The server hands the
template tool-call arguments as objects and its own JSON printer for `tojson`; both are reproduced here.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_run, parse_arguments, render_prompt, tojson  # noqa: E402


def main():
    import jinja2

    env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False)
    env.filters["tojson"] = lambda v, **kw: tojson(v)

    def raise_exception(msg):
        raise ValueError(msg)

    tpl = env.from_string(open(os.environ["TEMPLATE"], encoding="utf-8").read())
    n = bad = 0
    for name, path in (r.split("=", 1) for r in os.environ["RUNS"].split()):
        for r in load_run(path, name):
            body = json.loads(json.dumps(r["body"]))
            for m in body["messages"]:
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", tc)
                    fn["arguments"] = parse_arguments(fn.get("arguments"))
            want = tpl.render(messages=body["messages"], tools=body.get("tools"), add_generation_prompt=True,
                              raise_exception=raise_exception)
            got, _ = render_prompt(r["body"])
            n += 1
            if want != got:
                bad += 1
                if bad <= 3:
                    k = next((i for i, (a, b) in enumerate(zip(want, got)) if a != b), min(len(want), len(got)))
                    print("differs: %s at character %d (template %r, hand-written %r)" % (
                        r["id"], k, want[k - 20:k + 20], got[k - 20:k + 20]))
    print("%d requests, %d differ" % (n, bad))


if __name__ == "__main__":
    main()
