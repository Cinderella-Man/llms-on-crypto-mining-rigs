"""Per-file licence scan of a source tree: which files an agentic workspace may contain.

A permissive repository can still hold files under other terms: vendored third-party code, a file copied from another
project with its own header, a data file with a Creative Commons notice, generated or binary files. This scan reads
every file and gives it one verdict, so a workspace is built from an explicit allow-list instead of "the whole tree".

Per file it records: size, sha256, text or binary, SPDX-License-Identifier and SPDX-FileCopyrightText tags, licence
families found by their text signatures (GPL, LGPL, AGPL, MPL, EPL, CDDL, Creative Commons, Apache-2.0, MIT, BSD,
ISC, HPND-style, zlib, Boost, Unlicense, public domain, PSF), restrictive markers ("proprietary", "do not
distribute"), copyright lines, whether it sits in a vendored folder (vendor, _vendor, third_party, extern, ...) or below
a folder with its own LICENSE/COPYING file. The verdict:

  ok                    no notice of its own, or only the tree's own licence (and holders listed with --holders)
  ok:notice             the tree's licence, with copyright lines of other holders (kept verbatim; listed in NOTICE)
  review:<families>     a permissive licence other than the tree's (needs its own entry and the owner's decision)
  exclude:<reason>      restricted (copyleft, non-commercial, proprietary), vendored, own-licence subfolder, binary,
                        larger than --max-bytes, or not UTF-8

usage: python license_scan.py TREE --license MIT --holders 'regex' [--allow MIT,BSD-3-Clause] [--max-bytes 262144]
                              [--exclude GLOB ...] --out PREFIX
  writes PREFIX.tsv (one row per file) and PREFIX.json (counts, families, holders, the allow-list's size and hash)
Standard library only.
"""
import argparse
import fnmatch
import hashlib
import json
import os
import re
import sys

FAMILIES = [   # (family, signature regex); first match per family counts
    ("AGPL", r"GNU Affero General Public License|\bAGPL-?[0-9]"),
    ("LGPL", r"GNU (?:Lesser|Library) General Public License|\bLGPL-?[0-9]"),
    ("GPL", r"GNU General Public License|\bGPL-?[0-9]|GPLv[0-9]"),
    ("MPL", r"Mozilla Public License|\bMPL-?[12]"),
    ("EPL", r"Eclipse Public License|\bEPL-?[12]"),
    ("CDDL", r"Common Development and Distribution License|\bCDDL"),
    ("CC-NC", r"creativecommons\.org/licenses/[a-z-]*nc|Creative Commons[^\n]{0,40}Non-?Commercial|\bCC-BY-NC"),
    ("CC", r"Creative Commons|creativecommons\.org|\bCC-BY|\bCC0"),
    ("SSPL-BUSL", r"Server Side Public License|Business Source License|Commons Clause|Elastic License"),
    ("Apache-2.0", r"Apache License,?\s+Version 2\.0|apache\.org/licenses/LICENSE-2\.0"),
    ("MIT", r"Permission is hereby granted, free of charge, to any person obtaining"),
    ("BSD", r"Redistribution and use in source and binary forms"),
    ("ISC", r"Permission to use, copy, modify, and(?:/or)? distribute this software for any purpose with or without fee"),
    ("HPND", r"Permission to use, copy, modify,? and distribute this (?:\w+ )?software and its (?:\w+ )?documentation "
             r"for any purpose"),
    ("Zlib", r"This software is provided ['‘]as-is['’], without any express or implied"),
    ("BSL-1.0", r"Boost Software License"),
    ("Unlicense", r"This is free and unencumbered software released into the public domain"),
    ("public-domain", r"\bpublic domain\b"),
    ("PSF", r"PYTHON SOFTWARE FOUNDATION LICENSE|Python Software Foundation License|PSF License"),
]
RESTRICTED = {"AGPL", "LGPL", "GPL", "MPL", "EPL", "CDDL", "CC-NC", "SSPL-BUSL", "proprietary"}
PROPRIETARY = re.compile(r"\bproprietary\b|\bconfidential\b|do not (?:re)?distribute|not for (?:re)?distribution|"
                         r"unauthori[sz]ed copying", re.I)
SPDX = re.compile(r"SPDX-License-Identifier:\s*([A-Za-z0-9.+\-() ]+?)\s*(?:\*/|-->|#|$)", re.M)
SPDX_COPY = re.compile(r"SPDX-FileCopyrightText:\s*(.+?)\s*(?:\*/|-->|$)", re.M)
COPYRIGHT = re.compile(r"^[\s#*/;!-]*(?:Copyright\b|\(c\)\s|©)\s*(.{0,120})", re.M | re.I)
VENDORED = {"vendor", "vendored", "_vendor", "third_party", "thirdparty", "third-party", "3rdparty", "extern",
            "external", "deps", "contrib", "node_modules", "bundled", "vendors"}
LICENSE_FILE = re.compile(r"^(?:LICEN[CS]E|COPYING|NOTICE|UNLICENSE)(?:[-_][\w.-]*)?(?:\.(?:txt|md|rst))?$", re.I)
SPDX_FAMILY = {"MIT": "MIT", "MIT-0": "MIT", "Apache-2.0": "Apache-2.0", "BSD-2-Clause": "BSD", "BSD-3-Clause": "BSD",
               "0BSD": "BSD", "ISC": "ISC", "PSF-2.0": "PSF", "Python-2.0": "PSF", "Zlib": "Zlib", "BSL-1.0": "BSL-1.0",
               "Unlicense": "Unlicense", "CC0-1.0": "CC"}


def normalize(text):
    """Comment markers at line starts removed and whitespace collapsed, so that a licence text wrapped in # or *
    comments matches its signature."""
    text = re.sub(r"(?m)^[ \t]*(?:#+|//+|/\*+|\*+/?|;+|--|!|\"\"\"|''')?[ \t]?", "", text)
    return re.sub(r"\s+", " ", text)


def families_in(text):
    text = normalize(text)
    found = []
    for fam, rx in FAMILIES:
        if re.search(rx, text, re.I):
            if fam == "CC" and "CC-NC" in found:
                continue
            if fam == "GPL" and ("LGPL" in found or "AGPL" in found):
                # "GNU Lesser General Public License" also matches the GPL pattern's text; count GPL only when it
                # appears on its own
                if not re.search(r"(?<!Lesser )(?<!Library )(?<!Affero )GNU General Public License", text):
                    continue
            found.append(fam)
    if PROPRIETARY.search(text):
        found.append("proprietary")
    return found


def spdx_families(expr):
    out = set()
    for tok in re.split(r"[\s()]+", expr):
        if not tok or tok.upper() in ("AND", "OR", "WITH"):
            continue
        tok = tok.removesuffix("+").removesuffix("-only").removesuffix("-or-later")
        fam = SPDX_FAMILY.get(tok)
        if fam is None:
            for f in RESTRICTED:
                if tok.upper().startswith(f.split("-")[0]):
                    fam = f
                    break
        out.add(fam or f"LicenseRef:{tok}")
    return out


def scan_file(path, rel, args, own_license_dirs):
    st = os.stat(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        data = f.read()
    h.update(data)
    row = {"path": rel, "bytes": st.st_size, "sha256": h.hexdigest(), "kind": "text", "families": [], "spdx": "",
           "holders": [], "verdict": "ok"}
    parts = rel.split("/")
    if b"\x00" in data[:8192]:
        row["kind"] = "binary"
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            row["kind"] = "non-utf8"
            text = data.decode("latin-1")
        head = text[:16384] + "\n" + text[-4096:]
        fams = set(families_in(head))
        spdx = SPDX.findall(text[:16384])
        if spdx:
            row["spdx"] = " ; ".join(sorted(set(s.strip() for s in spdx)))
            for s in spdx:
                fams |= spdx_families(s)
        row["families"] = sorted(fams)
        holders = [m.strip() for m in COPYRIGHT.findall(head)] + [m.strip() for m in SPDX_COPY.findall(text[:16384])]
        row["holders"] = sorted(set(h for h in holders if h and not re.fullmatch(r"[\W\d]*", h)))[:8]
    fams = set(row["families"])
    restricted = sorted(fams & RESTRICTED)
    if row["kind"] == "binary":
        row["verdict"] = "exclude:binary"
    elif row["kind"] == "non-utf8":
        row["verdict"] = "exclude:not-utf8"
    elif st.st_size > args.max_bytes:
        row["verdict"] = "exclude:large"
    elif any(p.lower() in VENDORED for p in parts[:-1]):
        row["verdict"] = "exclude:vendored"
    elif any(rel.startswith(d + "/") for d in own_license_dirs):
        row["verdict"] = "exclude:own-licence-folder"
    elif len(parts) == 1 and LICENSE_FILE.match(parts[-1]):
        row["verdict"] = "ok:licence-file"   # the tree's own licence text (it may name other licences it lists)
    elif restricted:
        row["verdict"] = "exclude:" + "+".join(restricted)
    elif LICENSE_FILE.match(parts[-1]):
        row["verdict"] = "ok:licence-file" if (not fams or fams <= set(args.root)) else "review:" + "+".join(sorted(fams))
    else:
        other = sorted(f for f in fams - set(args.root) if f != "public-domain" or args.allow_pd)
        if other:
            row["verdict"] = "review:" + "+".join(other)
        elif args.holders and any(not re.search(args.holders, h, re.I) for h in row["holders"]):
            row["verdict"] = "ok:notice"
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tree")
    ap.add_argument("--license", required=True, help="the tree's own licence family (MIT, BSD, Apache-2.0, PSF, ...); "
                    "several with commas when the tree's licence text names several (e.g. PSF,HPND for CPython)")
    ap.add_argument("--holders", default="", help="regex of the tree's own copyright holders")
    ap.add_argument("--max-bytes", type=int, default=262144)
    ap.add_argument("--exclude", nargs="*", default=[], help="glob patterns (relative paths) left out of the scan")
    ap.add_argument("--allow-pd", action="store_true", help="treat a 'public domain' mention as no licence of its own")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.root = [x.strip() for x in args.license.split(",")]
    tree = os.path.abspath(args.tree)
    own_license_dirs = set()
    files = []
    for d, subdirs, names in os.walk(tree):
        subdirs[:] = sorted(s for s in subdirs if s not in (".git", "__pycache__"))
        rel_d = os.path.relpath(d, tree)
        if rel_d != ".":
            # a folder with a licence file of its own: excluded unless that licence is the tree's own family
            for n in names:
                if LICENSE_FILE.match(n) and n.upper().startswith(("LICEN", "COPYING", "UNLICEN")):
                    try:
                        fams = set(families_in(open(os.path.join(d, n), encoding="utf-8", errors="replace").read()))
                    except OSError:
                        fams = {"unreadable"}
                    if not fams or not fams <= set(args.root):
                        own_license_dirs.add(rel_d.replace(os.sep, "/"))
        for n in sorted(names):
            rel = os.path.normpath(os.path.join(rel_d, n)).replace(os.sep, "/")
            if any(fnmatch.fnmatchcase(rel, g) for g in args.exclude) or os.path.islink(os.path.join(d, n)):
                continue
            files.append((os.path.join(d, n), rel))
    rows = [scan_file(p, r, args, own_license_dirs) for p, r in files]
    with open(args.out + ".tsv", "w") as f:
        f.write("path\tbytes\tsha256\tkind\tfamilies\tspdx\tholders\tverdict\n")
        for r in rows:
            f.write("\t".join([r["path"], str(r["bytes"]), r["sha256"], r["kind"], ",".join(r["families"]), r["spdx"],
                               " | ".join(h.replace("\t", " ") for h in r["holders"]), r["verdict"]]) + "\n")
    counts, fams, holders = {}, {}, {}
    for r in rows:
        k = r["verdict"].split(":")[0] if r["verdict"].startswith("ok") else r["verdict"]
        counts[k] = counts.get(k, 0) + 1
        for f_ in r["families"]:
            fams[f_] = fams.get(f_, 0) + 1
        for h_ in r["holders"]:
            holders[h_] = holders.get(h_, 0) + 1
    allowed = sorted(r["path"] for r in rows if r["verdict"].startswith("ok"))
    summary = {"tree": tree, "license": args.root, "files": len(rows), "bytes": sum(r["bytes"] for r in rows),
               "verdicts": dict(sorted(counts.items(), key=lambda x: -x[1])),
               "families": dict(sorted(fams.items(), key=lambda x: -x[1])),
               "own_licence_folders": sorted(own_license_dirs),
               "top_holders": dict(sorted(holders.items(), key=lambda x: -x[1])[:40]),
               "allowed_files": len(allowed),
               "allowed_sha256": hashlib.sha256("\n".join(allowed).encode()).hexdigest()}
    with open(args.out + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("files", "verdicts", "families", "allowed_files")}))


if __name__ == "__main__":
    sys.exit(main())
