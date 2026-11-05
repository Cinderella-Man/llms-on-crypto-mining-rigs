#!/usr/bin/env python3
"""make_requests.py: the fixed greedy requests of C528 (Q1017), one set sent unchanged to every drafter's load.

  python3 -I make_requests.py --first FIRST_TURN.json --smoke smoke_greedy.json --text AGENT_TEXT --out REQ
      [--first-max-tokens 512] [--windows 0,6000,12000] [--window-bytes 6000] [--n-predict 384]

Writes into REQ, in the order rig_bits.sh sends them (the file name says the endpoint):
  1_first_turn.chat.json        the agent check's first-turn request (bench/agent_check/first_turn_request.sh's body,
                                ~2,760 tokens), made greedy: temperature 0, top_k 1, no top_p or min_p, max_tokens
                                --first-max-tokens
  2_smoke_greedy.chat.json      Q561's greedy smoke request, as it is (docs/data/2026-11-03_flash_next_dflash2/run/)
  3_cont_a.completion.json ...  one /completion per window of the agent text: --window-bytes bytes from each offset in
                                --windows, the start moved forward to the next line and the end back to the last line
                                break, sent as a raw prompt and continued greedily for --n-predict tokens
  manifest.tsv                  name, endpoint, prompt bytes, the window's byte range, md5 of the body
Standard library only; the text is not copied anywhere else.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=Path, required=True)
    ap.add_argument("--smoke", type=Path, required=True)
    ap.add_argument("--text", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--first-max-tokens", type=int, default=512)
    ap.add_argument("--windows", default="0,6000,12000")
    ap.add_argument("--window-bytes", type=int, default=6000)
    ap.add_argument("--n-predict", type=int, default=384)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []

    def put(name, endpoint, body, extra=""):
        data = json.dumps(body, ensure_ascii=False).encode()
        (a.out / f"{name}.{endpoint}.json").write_bytes(data)
        rows.append(f"{name}\t{endpoint}\t{len(data)}\t{extra}\t{hashlib.md5(data).hexdigest()}")

    first = json.loads(a.first.read_text())
    for k in ("top_p", "min_p"):
        first.pop(k, None)
    first.update(temperature=0, top_k=1, max_tokens=a.first_max_tokens)
    put("1_first_turn", "chat", first)
    put("2_smoke_greedy", "chat", json.loads(a.smoke.read_text()))
    text = a.text.read_bytes()
    for i, off in enumerate(int(x) for x in a.windows.split(",")):
        s = off if off == 0 else text.index(b"\n", off) + 1
        e = text.rindex(b"\n", s, min(len(text), s + a.window_bytes)) + 1
        prompt = text[s:e].decode("utf-8")
        put(f"{3 + i}_cont_{chr(ord('a') + i)}", "completion",
            {"prompt": prompt, "n_predict": a.n_predict, "temperature": 0, "top_k": 1, "stream": False},
            f"{s}-{e}")
    (a.out / "manifest.tsv").write_text("name\tendpoint\tbody_bytes\ttext_bytes\tmd5\n" + "\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
