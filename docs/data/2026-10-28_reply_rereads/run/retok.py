"""Do Flint's sampled tokens survive a text round trip? (Q260)

    python3 -I retok.py --turns turns.jsonl --server http://127.0.0.1:18431 --out retok.tsv [--examples ex.jsonl]

turns.jsonl is an agentic data export (bench/agentic_data/export.py, train/turns.jsonl): one row per turn with the
exact sampled ids in gen_tokens. Each turn's ids are detokenized (/detokenize) and the text tokenized again as a chat
prompt is (/tokenize, parse_special on). A turn whose ids differ was sampled with a non-canonical split somewhere: the
same text, other tokens. A server that re-tokenizes the conversation then finds its cached tokens only up to that
point. The row records where the first difference is (token index, distance from the end) and in which part of the
reply it falls (reasoning, the text after it, a tool call).
"""
import argparse
import json
import urllib.request


def post(server, path, obj):
    req = urllib.request.Request(server + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:18431")
    ap.add_argument("--out", required=True)
    ap.add_argument("--examples", help="JSON lines: the pieces at each first difference (generated against re-tokenized)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    ids_of = lambda s: post(a.server, "/tokenize", {"content": s, "add_special": False, "parse_special": True})["tokens"]
    special = {mk: ids_of(mk) for mk in ("<think>", "</think>", "<tool_call>", "</tool_call>")}
    special = {k: v[0] for k, v in special.items() if len(v) == 1}
    out = open(a.out, "w")
    out.write("session\tturn\tn_gen\tn_retok\tfirst_diff\tdiff_from_end\tpart\tn_reasoning\n")
    ex = open(a.examples, "w") if a.examples else None
    n = 0
    with open(a.turns) as f:
        for line in f:
            r = json.loads(line)
            gen = r["gen_tokens"]
            text = post(a.server, "/detokenize", {"tokens": gen})["content"]
            re_ids = ids_of(text)
            i = 0
            while i < min(len(gen), len(re_ids)) and gen[i] == re_ids[i]:
                i += 1
            same = i == len(gen) == len(re_ids)
            th_end = gen.index(special["</think>"]) if special.get("</think>") in gen else -1
            tc = gen.index(special["<tool_call>"]) if special.get("<tool_call>") in gen else -1
            if same:
                part, fd, dfe = "", -1, -1
            else:
                fd, dfe = i, len(gen) - i
                if th_end >= 0 and i < th_end:
                    part = "reasoning"
                elif tc >= 0 and i >= tc:
                    part = "tool-call"
                else:
                    part = "text"
            out.write(f"{r.get('session_id')}\t{r.get('turn')}\t{len(gen)}\t{len(re_ids)}\t{fd}\t{dfe}\t{part}\t"
                      f"{th_end if th_end >= 0 else ''}\n")
            if ex and not same:
                g = [post(a.server, "/detokenize", {"tokens": [t]})["content"] for t in gen[max(0, i - 3):i + 4]]
                c = [post(a.server, "/detokenize", {"tokens": [t]})["content"] for t in re_ids[max(0, i - 3):i + 4]]
                ex.write(json.dumps({"session": r.get("session_id"), "turn": r.get("turn"), "at": i, "part": part,
                                     "generated": g, "retokenized": c}) + "\n")
            n += 1
            if a.limit and n >= a.limit:
                break
    out.close()


if __name__ == "__main__":
    main()
