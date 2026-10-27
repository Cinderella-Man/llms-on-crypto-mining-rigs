"""Held-out agent sessions for scoring the drafter: every session of a fixed share of the modules, never trained on.

A stage-2 session repairs a module whose tests a stage-1 session wrote, so sessions of one module share code, tests
and wording; holding out whole modules keeps the test set's text out of training. A module is held out when the
SHA-256 of its workspace name falls in the first 1/--every of the hash space, so the choice depends only on the name
and stays the same when more sessions arrive. --take N keeps a random N of the held-out sessions for scoring (the
others are still left out of training).

usage: python heldout.py sessions_seq.jsonl [...] --test OUT_IDS --train OUT_IDS [--test-rows OUT.jsonl]
                         [--every 20] [--take 40] [--seed 1]
writes one session row id (agentic_N) per line: the scoring set and the training set; --test-rows also the scoring
set's rows (for feats.py seqs and score.py)
"""
import argparse
import hashlib
import json
import random
import re

S1 = re.compile(r"^s1-(.+)-wt(?:-a\d+)?(?:-r\d+)?$")
S2 = re.compile(r"^s2[a-z0-9]*-(.+)-a\d+(?:-r\d+)?-(?:fix_bug|implement)-\d+(?:-r\d+)?$")   # tasks.py --prefix


def module_of(session_id):
    m = S1.match(session_id) or S2.match(session_id)
    if not m:
        raise SystemExit(f"unknown session id {session_id}")
    return m[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seqs", nargs="+")
    ap.add_argument("--test", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test-rows")
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--take", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rows = [json.loads(line) for path in a.seqs for line in open(path)]
    held = lambda m: int(hashlib.sha256(m.encode()).hexdigest(), 16) % a.every == 0   # noqa: E731
    test = [r for r in rows if held(module_of(r["session_id"]))]
    train = [r for r in rows if not held(module_of(r["session_id"]))]
    if a.take and len(test) > a.take:
        test = sorted(random.Random(a.seed).sample(test, a.take), key=lambda r: int(r["id"].split("_")[-1]))
    with open(a.test, "w") as f:
        f.write("".join(r["id"] + "\n" for r in test))
    with open(a.train, "w") as f:
        f.write("".join(r["id"] + "\n" for r in train))
    if a.test_rows:
        with open(a.test_rows, "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in test))
    n_gen = lambda rs: sum(e - s for r in rs for s, e in r["gen_ranges"])   # noqa: E731
    mods = sorted({module_of(r["session_id"]) for r in rows if held(module_of(r["session_id"]))})
    print(f"held out {len(mods)} of {len({module_of(r['session_id']) for r in rows})} modules: {', '.join(mods)}")
    print(f"scoring set {len(test)} sessions, {n_gen(test)} generated tokens; training set {len(train)} sessions, "
          f"{n_gen(train)} generated tokens")


if __name__ == "__main__":
    main()
