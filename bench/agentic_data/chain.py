"""Walks a gate record's provenance chain back to its first stage and checks it as it stands now: every record passed,
every parent record's sha256 is the one its child recorded, and every file a record names is unchanged (size and
sha256). Gates check their direct parents when they run; this re-checks the whole chain later, e.g. before a store or
a head is published.

Records are those of gate.py (workspaces, tasks, sessions, answers, dataset) and check_sources.py (prompts, answers,
dump, derived). Prints one line per record (stage, verdict, files, source ids, writers, held-out test files) and a
final verdict.

usage: python chain.py RECORD.gate.json [--no-files] [--out CHAIN.json]
  --no-files  skip re-hashing the records' files (fast; only the records and their links are checked)
"""
import argparse
import hashlib
import json
import os
import sys


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record")
    ap.add_argument("--no-files", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    todo, seen, rows, errors = [(os.path.abspath(a.record), None, 0)], set(), [], []
    while todo:
        path, want, depth = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        if not os.path.exists(path):
            errors.append(f"{path}: missing")
            continue
        raw = open(path, "rb").read()
        got = hashlib.sha256(raw).hexdigest()
        if want and got != want:
            errors.append(f"{path}: sha256 {got[:12]} is not the {want[:12]} its child recorded")
        rec = json.loads(raw)
        if rec.get("verdict") != "PASS":
            errors.append(f"{path}: verdict {rec.get('verdict')}")
        n_files = 0
        for f in rec.get("files", []):
            n_files += 1
            if a.no_files:
                continue
            p = f["path"]
            if os.path.isdir(p):
                continue    # a folder's own record hashes its contents; workspaces are checked by their gate
            if not os.path.exists(p):
                errors.append(f"{path}: file {p} missing")
            elif os.path.getsize(p) != f.get("bytes", os.path.getsize(p)) or \
                    ("sha256" in f and sha256_file(p) != f["sha256"]):
                errors.append(f"{path}: file {p} changed after it was gated")
        held = {k: (v or {}).get("path") for k, v in (rec.get("heldout") or {}).items() if v}
        rows.append({"depth": depth, "path": path, "sha256": got, "gate": rec.get("gate"), "stage": rec.get("stage"),
                     "kind": rec.get("kind"), "verdict": rec.get("verdict"), "files": n_files,
                     "source_ids": rec.get("source_ids", []), "writers": rec.get("writers", []), "heldout": held,
                     "sources_json": (rec.get("sources_json") or {}).get("sha256")})
        for p in rec.get("parents", []):
            todo.append((p["path"], p.get("sha256"), depth + 1))
    for r in sorted(rows, key=lambda r: (r["depth"], r["path"])):
        extra = f" held out: {', '.join(r['heldout'])}" if r["heldout"] else ""
        print(f"{'  ' * r['depth']}{r['stage']}{'/' + r['kind'] if r['kind'] else ''} {r['verdict']} "
              f"{r['path']} ({r['files']} files; sources {', '.join(r['source_ids']) or '-'}; "
              f"writers {', '.join(r['writers']) or '-'}){extra}")
    srcs = sorted({r["sources_json"] for r in rows if r["sources_json"]})
    print(f"{len(rows)} records, sources.json versions {len(srcs)}; "
          f"{'PASS' if not errors else 'FAIL'}{': ' + '; '.join(errors[:10]) if errors else ''}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"records": rows, "errors": errors, "verdict": "FAIL" if errors else "PASS"}, f, indent=1)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
