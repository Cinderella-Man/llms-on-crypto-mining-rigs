"""The data gate of the staircase drafter (Q20): every training file must trace back to sources.json.

Each pipeline stage runs this gate on what it wrote before the next stage may read it: prompts -> answers -> dumps ->
vocabulary, packs, heads (and the n-gram store). A run checks every row, writes a manifest (one TSV row per prompt,
answer or dumped record) and a gate record (JSON: verdict, the files' sizes and hashes, sources.json's hash, the source
ids and writers involved, and the path and sha256 of each parent gate record), and exits 1 on any failure. A later
stage names its inputs' gate records with --parent; the gate checks that they passed, that their files are unchanged
and that their sources are still approved, and stores their sha256, so every file's history chains back to the prompts.

  prompts  a prompt file. Each row names its source (source_id, row_id; task_source for our own task texts). Source
           files: the path must belong to that source under sources.json's path rules and exist in the tree (--tree),
           and the file's text (or its path) must appear in the prompt. Data set rows: the row must exist in the cached
           parquet shard (--hf-cache), satisfy the source's match conditions and carry exactly its text. No assistant
           text, and no 12-word span shared with a Q51 prompt (--q51), nor a file Q51 is built from (--q51-sources).
  answers  answer files (generate.py rows). Each answer's prompt is in a passed prompt file, its writer is approved,
           its prompt tokens are the prompt's own (re-tokenized with --tokenizer) and its decoded text shares no
           12-word span with a Q51 prompt.
  dump     feature files (llama-staircase-dump records). Each record is a passed answer, token for token.
  derived  anything built from gated files (--kind vocab, pack, head, store, model, other): --cmd is the exact command
           that wrote it; every file it names must be a parent's file, an approved Flint model file, a test file named
           with --test (never trained on) or an id list whose ids all belong to the parents.

usage: python check_sources.py SOURCES.json prompts P.jsonl --q51 Q51.jsonl [--q51-sources S.jsonl]
                                                           --tree NAME=DIR [...] [--hf-cache DIR]
       python check_sources.py SOURCES.json answers G.jsonl [...] --parent P.gate.json [...] --q51 Q51.jsonl
                                                           --tokenizer FLINT_HF_DIR
       python check_sources.py SOURCES.json dump F.bin [...] --parent G.gate.json [...]
       python check_sources.py SOURCES.json derived OUT [...] --kind KIND --parent X.gate.json [...] --cmd "COMMAND"
                                                           [--test FILE ...] [--allow-dir DIR ...]
  --out GATE.json   the gate record (default: the first file + .gate.json); the manifest is written next to it (.tsv)
Run with Python 3.12 (sources.json's python-stdlib tree): its sys.stdlib_module_names decides what is standard library.
"""
import argparse
import collections
import fnmatch
import glob
import hashlib
import json
import os
import shlex
import sys
import time

STDLIB = set(sys.stdlib_module_names) | {"idlelib", "lib2to3", "turtledemo", "ensurepip"}
FULL_HASH_MAX = 4 << 30   # bigger files get a sampled hash (64 x 1 MiB and the tail), recorded as such
SCRIPT_EXT = (".py", ".sh")


def load_sources(path):
    raw = open(path, "rb").read()
    src = json.loads(raw)
    src["sha256"] = hashlib.sha256(raw).hexdigest()
    src["path"] = os.path.abspath(path)
    src["by_id"] = {s["id"]: s for s in src["sources"]}
    src["writer_ids"] = {w["id"] for w in src["writers"]}
    return src


def source_of(src, tree, rel):
    """The id of the source a file at path rel (relative to the tree's root) belongs to, or None. The entry with the
    longest matching path wins; its excludes and the tree's apply, except to a file it names exactly."""
    rel = rel.replace(os.sep, "/")
    if any(fnmatch.fnmatchcase(rel, p) for p in src["trees"][tree].get("exclude", [])):
        return None
    best, blen, exact = None, -1, False
    for s in src["sources"]:
        if s.get("tree") != tree:
            continue
        for p in s["paths"]:
            if p == "@stdlib-modules":
                ok, n, ex = rel.split("/")[0].removesuffix(".py") in STDLIB, 0, False
            elif p == "" or p.endswith("/"):
                ok, n, ex = rel.startswith(p), len(p), False
            else:
                ok, n, ex = rel == p, len(p), True
            if ok and n > blen:
                best, blen, exact = s, n, ex
    if best is None or (not exact and any(fnmatch.fnmatchcase(rel, p) for p in best.get("exclude", []))):
        return None
    return best["id"]


def file_id(path):
    """Size and hash of a file (sampled above FULL_HASH_MAX) or of a folder (its files' relative paths and hashes)."""
    path = os.path.abspath(path)
    if os.path.isdir(path):
        h = hashlib.sha256()
        n = 0
        for d, subdirs, names in sorted(os.walk(path)):
            subdirs.sort()
            for name in sorted(names):
                f = file_id(os.path.join(d, name))
                h.update(f"{os.path.relpath(os.path.join(d, name), path)}\0{f.get('sha256') or f['sampled_sha256']}\0"
                         .encode())
                n += f["bytes"]
        return {"path": path, "bytes": n, "dir_sha256": h.hexdigest()}
    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if size <= FULL_HASH_MAX:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
            return {"path": path, "bytes": size, "sha256": h.hexdigest()}
        for i in range(64):
            f.seek(i * (size // 64))
            h.update(f.read(1 << 20))
        f.seek(size - (1 << 20))
        h.update(f.read())
    return {"path": path, "bytes": size, "sampled_sha256": h.hexdigest()}


def words(messages):
    return " ".join((m.get("content") or "") for m in messages).lower().split()


def q51_spans(path):
    spans = set()
    for line in open(path):
        w = words(json.loads(line)["messages"])
        spans |= {" ".join(w[i:i + 12]) for i in range(len(w) - 11)}
    return spans


def shares_12(w, spans):
    return any(" ".join(w[i:i + 12]) in spans for i in range(len(w) - 11))


def load_parent(path, src, errors):
    """A parent gate record: it passed, its files are unchanged and its sources and writers are still approved."""
    raw = open(path, "rb").read()
    rec = json.loads(raw)
    if rec.get("verdict") != "PASS":
        errors.append(f"parent {path}: verdict {rec.get('verdict')}")
    for f in rec["files"]:
        if not os.path.exists(f["path"]):
            errors.append(f"parent {path}: {f['path']} is missing")
        elif file_id(f["path"]) != f:
            errors.append(f"parent {path}: {f['path']} changed after it was gated")
    for sid in rec.get("source_ids", []):
        if sid not in src["by_id"]:
            errors.append(f"parent {path}: source {sid} is no longer approved")
    for w in rec.get("writers", []):
        if w not in src["writer_ids"]:
            errors.append(f"parent {path}: writer {w} is no longer approved")
    return rec, {"path": os.path.abspath(path), "sha256": hashlib.sha256(raw).hexdigest(), "stage": rec["stage"]}


def parent_rows(rec):
    """{row id: manifest row} of a parent gate record's manifest."""
    if not rec.get("manifest"):
        return {}
    lines = open(rec["manifest"]["path"]).read().splitlines()
    head = lines[0].split("\t")
    return {r[0]: dict(zip(head, r)) for r in (x.split("\t") for x in lines[1:])}


def hf_shard(cache, repo, shard):
    """The cached parquet file prompts.shard() downloads: the first shard of the one cached snapshot."""
    snaps = glob.glob(os.path.join(cache, "datasets--" + repo.replace("/", "--"), "snapshots", "*"))
    if len(snaps) != 1:
        raise SystemExit(f"{repo}: {len(snaps)} cached snapshots in {cache} (need exactly one)")
    files = sorted(glob.glob(os.path.join(snaps[0], shard, "*.parquet")))
    return files[0], os.path.basename(snaps[0])


def stage_prompts(args, src, errors, out_rows, rec):
    spans = q51_spans(args.q51)
    banned = {os.path.basename(json.loads(x).get("source", "")) for x in open(args.q51_sources)} - {""} \
        if args.q51_sources else set()
    trees = dict(t.split("=", 1) for t in args.tree or [])
    py = src["trees"]["python-stdlib"].get("python")
    if py and f"{sys.version_info[0]}.{sys.version_info[1]}" != py:
        raise SystemExit(f"run with Python {py} (the python-stdlib tree's version), not {sys.version.split()[0]}")
    datasets, tot, seen, text_cache = {}, collections.Counter(), set(), {}
    own = {s["id"] for s in src["sources"] if s["kind"] == "own"}
    rec["datasets"] = {}
    for path in args.files:
        for n, line in enumerate(open(path), 1):
            p = json.loads(line)
            pid, sid, rid = p.get("id", f"{path}:{n}"), p.get("source_id"), p.get("row_id")
            bad = []
            if pid in seen:
                bad.append("duplicate id")
            seen.add(pid)
            s = src["by_id"].get(sid)
            if s is None:
                bad.append(f"source {sid!r} not approved")
            elif s["kind"] == "source-files":
                tree = s["tree"]
                if source_of(src, tree, rid or "") != sid:
                    bad.append(f"{rid} is not part of {sid} ({source_of(src, tree, rid or '')})")
                if tree not in trees:
                    bad.append(f"no --tree {tree}")
                else:
                    fp = os.path.join(trees[tree], rid)
                    if fp not in text_cache:
                        try:
                            text_cache[fp] = open(fp, encoding="utf-8").read()
                        except (OSError, UnicodeDecodeError):
                            text_cache[fp] = None
                    text = text_cache[fp]
                    msgs = " ".join((m.get("content") or "") + json.dumps(m.get("tool_calls") or "")
                                    for m in p["messages"])
                    if text is None:
                        bad.append(f"{rid} not in the {tree} tree")
                    elif p["category"] in ("edit", "agent") and text not in msgs and rid not in msgs:
                        bad.append(f"neither the text nor the path of {rid} is in the prompt")
                if os.path.basename(rid or "") in banned:
                    bad.append(f"{rid}: Q51 is built from this file")
                if p.get("task_source") not in own:
                    bad.append(f"task_source {p.get('task_source')!r} not approved")
            elif s["kind"] == "dataset":
                if not args.hf_cache:
                    bad.append("no --hf-cache to check the data set row")
                else:
                    if sid not in datasets:
                        import pyarrow.parquet as pq
                        f, snap = hf_shard(args.hf_cache, s["repo"], s["shard"])
                        datasets[sid] = {r[s["row_key"]]: r for r in pq.read_table(f).to_pylist()}
                        rec["datasets"][sid] = {"repo": s["repo"], "snapshot": snap, **file_id(f)}
                    r = datasets[sid].get(rid)
                    user = [m for m in p["messages"] if m["role"] == "user"]
                    if r is None:
                        bad.append(f"row {rid} not in {s['repo']}")
                    elif any(r.get(k) != v for k, v in s["match"].items()):
                        bad.append(f"row {rid} fails the match conditions")
                    elif len(p["messages"]) != 1 or len(user) != 1 or user[0]["content"] != r[s["text_field"]]:
                        bad.append(f"the prompt is not exactly row {rid}'s {s['text_field']}")
            else:
                bad.append(f"source {sid} has kind {s['kind']}: not a prompt source")
            if any(m["role"] == "assistant" and (m.get("content") or "").strip() for m in p["messages"]):
                bad.append("assistant text in the prompt")
            if shares_12(words(p["messages"]), spans):
                bad.append("shares a 12-word span with Q51")
            out_rows.append([pid, p.get("category", ""), sid or "", rid or "", p.get("task_source", ""),
                             s["licence"] if s else "", "; ".join(bad) or "ok"])
            tot[(sid, p.get("category"))] += 1
            errors += [f"{pid}: {b}" for b in bad]
    rec["source_ids"] = sorted({k[0] for k in tot if k[0] in src["by_id"]} | ({p for p in own} if tot else set()))
    rec["writers"] = []
    rec["summary"] = {f"{k[0]} / {k[1]}": v for k, v in sorted(tot.items(), key=lambda x: -x[1])}
    return ["prompt_id", "category", "source_id", "row_id", "task_source", "licence", "check"]


def stage_answers(args, src, errors, out_rows, rec, parents):
    spans = q51_spans(args.q51)
    prompts = {}
    for prec in parents:
        if prec["stage"] != "prompts":
            errors.append(f"parent of stage {prec['stage']}: answers come from prompt files")
            continue
        for f in prec["files"]:
            for line in open(f["path"]):
                p = json.loads(line)
                prompts[p["id"]] = p
    from transformers import AutoTokenizer
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from generate import prompt_ids_hf
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    tot, seen = collections.Counter(), set()
    for path in args.files:
        for n, line in enumerate(open(path), 1):
            try:
                a = json.loads(line)
            except ValueError:
                errors.append(f"{path}:{n}: not a JSON row")
                continue
            aid, bad = a.get("id"), []
            p = prompts.get(aid)
            if aid in seen:
                bad.append("duplicate answer")
            seen.add(aid)
            if p is None:
                bad.append("its prompt is not in a passed prompt file")
            else:
                for k in ("source_id", "row_id", "category"):
                    if a.get(k) != p.get(k):
                        bad.append(f"{k} {a.get(k)!r} differs from the prompt's {p.get(k)!r}")
                if a.get("prompt_tokens") != prompt_ids_hf(tok, p):
                    bad.append("prompt tokens are not the prompt's")
            if a.get("writer") not in src["writer_ids"]:
                bad.append(f"writer {a.get('writer')!r} not approved")
            if a.get("stop_type") not in ("eos", "limit"):
                bad.append(f"stop_type {a.get('stop_type')!r}")
            if shares_12(tok.decode(a.get("gen_tokens") or []).lower().split(), spans):
                bad.append("the answer shares a 12-word span with Q51")
            out_rows.append([aid or f"{path}:{n}", a.get("writer", ""), a.get("category", ""), a.get("source_id", ""),
                             str(len(a.get("prompt_tokens") or [])), str(len(a.get("gen_tokens") or [])),
                             os.path.basename(path), "; ".join(bad) or "ok"])
            tot[(a.get("source_id"), a.get("writer"), "n")] += 1
            tot[(a.get("source_id"), a.get("writer"), "t")] += len(a.get("gen_tokens") or [])
            errors += [f"{aid}: {b}" for b in bad]
    rec["source_ids"] = sorted({k[0] for k in tot if k[0] in src["by_id"]})
    rec["writers"] = sorted({k[1] for k in tot if k[1] in src["writer_ids"]})
    rec["summary"] = {f"{k[0]} / {k[1]}": {"answers": tot[k], "answer_tokens": tot[(k[0], k[1], "t")]}
                      for k in sorted(tot, key=lambda x: -tot[x]) if k[2] == "n"}
    return ["answer_id", "writer", "category", "source_id", "prompt_tokens", "answer_tokens", "file", "check"]


def stage_dump(args, src, errors, out_rows, rec, parents):
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import feats
    answers = {}
    for prec in parents:
        if prec["stage"] != "answers":
            errors.append(f"parent of stage {prec['stage']}: dumps come from answer files")
            continue
        for f in prec["files"]:
            for line in open(f["path"]):
                try:
                    a = json.loads(line)
                except ValueError:
                    continue
                toks = np.asarray(a["prompt_tokens"] + a["gen_tokens"], np.int32)
                num = int(a["id"].split("_")[-1])
                if num in answers:
                    errors.append(f"answer number {num} appears twice in the parents")
                answers[num] = (a["id"], len(a["prompt_tokens"]), hashlib.sha1(toks.tobytes()).hexdigest(),
                                a["writer"], a["source_id"])
    tot, seen = collections.Counter(), collections.Counter()
    for path in args.files:
        ff = feats.FeatFile(path)
        for i in range(len(ff)):
            r = ff[i]
            a, bad = answers.get(r.seq_id), []
            seen[r.seq_id] += 1
            if a is None:
                bad.append("not a passed answer")
            elif r.n_prompt != a[1] or hashlib.sha1(np.ascontiguousarray(r.tokens, np.int32).tobytes()).hexdigest() != a[2]:
                bad.append(f"tokens differ from answer {a[0]}")
            out_rows.append([a[0] if a else str(r.seq_id), a[3] if a else "", a[4] if a else "", str(r.n_gen),
                             os.path.basename(path), "; ".join(bad) or "ok"])
            if a:
                tot[(a[4], a[3], "n")] += 1
                tot[(a[4], a[3], "t")] += r.n_gen
            errors += [f"record {r.seq_id} in {path}: {b}" for b in bad]
    rec["duplicates"] = sum(1 for v in seen.values() if v > 1)
    rec["source_ids"] = sorted({k[0] for k in tot})
    rec["writers"] = sorted({k[1] for k in tot})
    rec["summary"] = {f"{k[0]} / {k[1]}": {"records": tot[k], "answer_tokens": tot[(k[0], k[1], "t")]}
                      for k in sorted(tot, key=lambda x: -tot[x]) if k[2] == "n"}
    return ["answer_id", "writer", "source_id", "answer_tokens", "file", "check"]


def model_file_ok(src, path):
    name, size = os.path.basename(path), os.path.getsize(path)
    for w in src["writers"]:
        if any(f["name"] == name and f["bytes"] == size for f in w.get("files", [])) \
                or any(fnmatch.fnmatchcase(name, g) for g in w.get("names", [])):
            return w["id"]
    return None


def stage_derived(args, src, errors, out_rows, rec, parents, outputs):
    if not args.cmd:
        raise SystemExit("derived: --cmd (the exact command that wrote the files) is required")
    files = {f["path"]: prec for prec in parents for f in prec["files"]}
    ids = {}
    for prec in parents:
        ids.update(parent_rows(prec))
    tests = {os.path.abspath(t) for t in args.test or []}
    allow = {os.path.abspath(d) for d in args.allow_dir or []}
    inputs = []
    for token in shlex.split(args.cmd):
        for t in token.split("=", 1)[-1].split(",") if "=" in token or "," in token else [token]:
            if not t or t.startswith("-") or not os.path.exists(t):
                continue
            ap, role = os.path.abspath(t), None
            if ap in outputs or ap.endswith(SCRIPT_EXT) or ap == src["path"]:
                continue
            if ap in files:
                role = f"parent ({files[ap]['stage']})"
            elif ap in tests:
                role = "test (not trained on)"
            elif os.path.isdir(ap):
                role = "output folder" if ap in allow else None
            elif ap.endswith(".gguf") and model_file_ok(src, ap):
                role = f"model ({model_file_ok(src, ap)})"
            elif os.path.getsize(ap) < 64 << 20:
                try:
                    lines = [x.split()[0] for x in open(ap) if x.strip()]
                    if lines and all(x in ids for x in lines):
                        role = f"id list ({len(lines)} ids of the parents)"
                except (UnicodeDecodeError, IndexError):
                    pass
            inputs.append({"path": ap, "role": role or "NOT TRACED"})
            out_rows.append([ap, role or "NOT TRACED"])
            if role is None:
                errors.append(f"input {ap} does not trace back to a parent, an approved model or a --test file")
    rec["kind"], rec["cmd"], rec["inputs"] = args.kind, args.cmd, inputs
    rec["source_ids"] = sorted({s for prec in parents for s in prec.get("source_ids", [])})
    rec["writers"] = sorted({w for prec in parents for w in prec.get("writers", [])})
    return ["input", "role"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources")
    ap.add_argument("stage", choices=("prompts", "answers", "dump", "derived"))
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--parent", nargs="*", default=[])
    ap.add_argument("--q51")
    ap.add_argument("--q51-sources")
    ap.add_argument("--tree", action="append")
    ap.add_argument("--hf-cache")
    ap.add_argument("--tokenizer")
    ap.add_argument("--kind", choices=("vocab", "pack", "head", "store", "model", "other"))
    ap.add_argument("--cmd")
    ap.add_argument("--test", nargs="*")
    ap.add_argument("--allow-dir", nargs="*")
    args = ap.parse_args()
    if args.stage in ("prompts", "answers") and not args.q51:
        raise SystemExit("--q51 (Q51's prompt file) is required")
    if args.stage == "answers" and not args.tokenizer:
        raise SystemExit("--tokenizer (Flint's Hugging Face folder) is required")
    if args.stage != "prompts" and not args.parent:
        raise SystemExit("--parent: the gate records of this stage's inputs are required")
    src = load_sources(args.sources)
    out = args.out or args.files[0].rstrip("/") + ".gate.json"
    errors, rows = [], []
    rec = {"gate": "check_sources.py", "stage": args.stage, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "sources_json": {"path": src["path"], "sha256": src["sha256"]}}
    parents = []
    for p in args.parent:
        prec, ref = load_parent(p, src, errors)
        parents.append(prec)
        rec.setdefault("parents", []).append(ref)
    if args.q51:
        rec["q51"] = file_id(args.q51)
    outputs = {os.path.abspath(f) for f in args.files}
    if args.stage == "prompts":
        head = stage_prompts(args, src, errors, rows, rec)
    elif args.stage == "answers":
        head = stage_answers(args, src, errors, rows, rec, parents)
    elif args.stage == "dump":
        head = stage_dump(args, src, errors, rows, rec, parents)
    else:
        head = stage_derived(args, src, errors, rows, rec, parents, outputs)
    rec["files"] = [file_id(f) for f in args.files]
    manifest = out.removesuffix(".json") + ".tsv"
    with open(manifest, "w") as f:
        f.write("\t".join(head) + "\n")
        for r in rows:
            f.write("\t".join(str(x).replace("\t", " ").replace("\n", " ") for x in r) + "\n")
    rec["manifest"] = file_id(manifest)
    rec["rows"] = len(rows)
    rec["failures"] = len(errors)
    rec["first_failures"] = errors[:50]
    rec["verdict"] = "FAIL" if errors else "PASS"
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    for k, v in (rec.get("summary") or {}).items():
        print(f"{k}: {v}")
    for e in errors[:20]:
        print("FAIL:", e)
    print(f"{rec['verdict']}: {args.stage}, {len(rows)} rows, {len(errors)} failures; gate record {out}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
