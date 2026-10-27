#!/usr/bin/env python3
"""Lists Hugging Face model repositories' files at their current revision, with sizes and LFS SHA-256, and writes a
manifest (one line per file: repo, revision, path, size, sha256) for scripts/hf_get.sh, which downloads exactly those
revisions and checks every file. A file is kept when its path matches the repo's regex (case-insensitive).
Standard library only.

usage: python3 -I scripts/hf_manifest.py manifest.tsv <org/repo>=<regex> [<org/repo>=<regex> ...]
e.g.   python3 -I scripts/hf_manifest.py manifest.tsv 'AtomicChat/Ornith-1.5-122B-GGUF=AD-IQ3_XXS-IQ2_S'"""
import json
import re
import sys
import urllib.request


def api(path):
    return json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{path}", timeout=60))


def main():
    out = sys.argv[1]
    rows = []
    for arg in sys.argv[2:]:
        repo, _, pattern = arg.partition("=")
        rev = api(repo)["sha"]
        tree = api(f"{repo}/tree/{rev}?recursive=true")
        files = [t for t in tree if t.get("type") == "file" and re.search(pattern, t["path"], re.I)]
        total = sum(t.get("size", 0) for t in files)
        print(f"== {repo} @ {rev[:12]}: {len(files)} files, {total / 1e9:.1f} GB")
        for t in files:
            sha = (t.get("lfs") or {}).get("oid", "")
            print(f"   {t['path']}  {t.get('size', 0) / 1e9:.2f} GB  {sha[:12] or '(git file)'}")
            rows.append("\t".join([repo, rev, t["path"], str(t.get("size", 0)), sha]))
    with open(out, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"manifest: {out} ({len(rows)} files)")


if __name__ == "__main__":
    main()
