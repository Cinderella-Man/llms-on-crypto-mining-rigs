"""Q20 STRICT (2026-10-25): the site-packages licence ruling: agent/edit answers whose source file comes from a
package licensed MIT, BSD (2- or 3-clause), Apache-2.0 or PSF are clean; copyleft (GPL/LGPL/AGPL/MPL/EPL, or a
licence combined with one), other licences (ISC, CC0, ...) and unclear files are not. Licences as each package's
installed METADATA states them (pip install huggingface_hub pyarrow in python:3.12-slim, the environment prompts.py
built the prompts in; versions as installed on 2026-10-25, listed by sitepkg_licences.py); pip's vendored packages and
pyarrow's vendored headers are judged one by one. Writes WORK_DIR/strict-sitepkgs.md (per package: version, licence,
decision, answers, answer tokens), adds the allowed answers to strict-ids.txt (from count_strict.py), and writes
strict-ids-notool.txt (without the generic tool-call answers, whose sources were under audit).

usage: python3 -I strict_sitepkgs.py WORK_DIR
  WORK_DIR holds the drafter's prompt files, answer files (gen-<label>.jsonl) and strict-ids.txt.
"""
import collections
import json
import sys

W = sys.argv[1]
OK = {"MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "PSF-2.0", "Apache-2.0 OR BSD-2-Clause"}
TOP = {   # package dir -> (distribution and version, licence from its METADATA)
    "pyarrow": ("pyarrow 26.0.0", "Apache-2.0"), "pip": ("pip 25.0.1", "MIT"),
    "huggingface_hub": ("huggingface_hub 2.3.0", "Apache-2.0"), "tqdm": ("tqdm 4.70.1", "MPL-2.0 AND MIT"),
    "fsspec": ("fsspec 2026.9.0", "BSD-3-Clause"), "httpcore2": ("httpcore2 2.16.0", "BSD-3-Clause"),
    "anyio": ("anyio 4.15.1", "MIT"), "httpx2": ("httpx2 2.16.0", "BSD-3-Clause"), "yaml": ("PyYAML 6.0.3", "MIT"),
    "_yaml": ("PyYAML 6.0.3", "MIT"), "packaging": ("packaging 26.3", "Apache-2.0 OR BSD-2-Clause"),
    "h11": ("h11 0.16.0", "MIT"), "idna": ("idna 3.20", "BSD-3-Clause"), "click": ("click 8.5.0", "BSD-3-Clause"),
    "filelock": ("filelock 4.2.4", "MIT"), "truststore": ("truststore 0.10.5", "MIT")}
PIP_VENDOR = {   # pip/_vendor/<name> -> licence of the vendored project (pip 25.0.1's vendor list)
    "rich": "MIT", "pygments": "BSD-2-Clause", "urllib3": "MIT", "requests": "Apache-2.0", "cachecontrol": "Apache-2.0",
    "packaging": "Apache-2.0 OR BSD-2-Clause", "truststore": "MIT", "resolvelib": "ISC", "msgpack": "Apache-2.0",
    "idna": "BSD-3-Clause", "platformdirs": "MIT", "distlib": "PSF-2.0", "certifi": "MPL-2.0", "distro": "Apache-2.0",
    "tomli": "MIT", "pyproject_hooks": "MIT", "__init__.py": "MIT (pip's own)"}
ARROW_VENDOR = {"double-conversion": "BSD-3-Clause", "datetime": "MIT", "datetime.h": "Apache-2.0 (Arrow's own)",
                "portable-snippets": "CC0-1.0", "strptime.h": "unclear (musl/NetBSD origin)"}


def classify(src):
    p = src.split("/")
    top = p[1]
    if top == "pip" and len(p) > 3 and p[2] == "_vendor":
        lic = PIP_VENDOR.get(p[3], "unknown")
        return f"pip/_vendor/{p[3]} (vendored in pip 25.0.1)", lic, lic.split(" (")[0] in OK
    if top == "pyarrow" and len(p) > 5 and p[2:5] == ["include", "arrow", "vendored"]:
        lic = ARROW_VENDOR.get(p[5], "unknown")
        return f"pyarrow/include/arrow/vendored/{p[5]}", lic, lic.split(" (")[0] in OK
    dist, lic = TOP.get(top, (top, "unknown"))
    return dist, lic, lic in OK


prompts, decision = {}, {}
for f in ("prompts.jsonl", "prompts-ext.jsonl", "prompts-ext2.jsonl", "prompts-tools.jsonl"):
    for line in open(f"{W}/{f}"):
        r = json.loads(line)
        prompts[r["id"]] = r["category"]
        s = r.get("source", "")
        if r["category"] in ("edit", "agent") and s.startswith("site-packages/"):
            decision[r["id"]] = classify(s)
n, t = collections.Counter(), collections.Counter()
allowed = []
for g in "A B B2 B3 C C2 C2s E1 E2 F1 F2 FV3 V1".split():
    for line in open(f"{W}/gen-{g}.jsonl"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        d = decision.get(r["id"])
        if d:
            n[d] += 1
            t[d] += len(r["gen_tokens"])
            if d[2]:
                allowed.append(r["id"])
with open(f"{W}/strict-sitepkgs.md", "w") as f:
    f.write("# site-packages sources of the Q20 drafter data (licence ruling of 2026-10-25)\n\n"
            "Agent and whole-file edit prompts built from source files of packages installed in the Python image. Licence "
            "as the package's METADATA states it (versions as installed when checked, 2026-10-25); pip's vendored packages "
            "and pyarrow's vendored headers judged one by one. Allowed: MIT, BSD-2/3-Clause, Apache-2.0, PSF.\n\n"
            "| Package | Licence | Included | Answers | Answer tokens |\n|---|---|---|---|---|\n")
    for d, k in sorted(t.items(), key=lambda x: -x[1]):
        f.write(f"| {d[0]} | {d[1]} | {'yes' if d[2] else 'no'} | {n[d]} | {k / 1e6:.3f} M |\n")
    inc = sum(k for d, k in t.items() if d[2])
    f.write(f"\nIncluded: {sum(n[d] for d in n if d[2])} answers, {inc / 1e6:.2f} M answer tokens; excluded: "
            f"{sum(n[d] for d in n if not d[2])} answers, {(sum(t.values()) - inc) / 1e6:.2f} M.\n")
ids = [x.strip() for x in open(f"{W}/strict-ids.txt") if x.strip()]
merged = list(dict.fromkeys(ids + allowed))
with open(f"{W}/strict-ids.txt", "w") as f:
    f.write("\n".join(merged) + "\n")
with open(f"{W}/strict-ids-notool.txt", "w") as f:
    f.write("\n".join(x for x in merged if prompts.get(x) != "tool") + "\n")
print(open(f"{W}/strict-sitepkgs.md").read())
print(f"strict-ids.txt: {len(ids)} -> {len(merged)} answers; strict-ids-notool.txt: "
      f"{sum(1 for x in merged if prompts.get(x) != 'tool')}")
