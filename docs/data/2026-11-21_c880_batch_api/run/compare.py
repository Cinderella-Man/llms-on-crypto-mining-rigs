#!/usr/bin/env python3
# C880 (2026-11-21): compares tools.sh's outputs of the base build and the head build.
#   exit codes   rc.tsv, run by run;
#   pipespec     every PIPESPEC line of results.txt (check_identity.sh, check_streams.sh), job by job, every field but
#                the timings (t_gen, tok_s, t_race, t_first, host_*_us, t_all, total_tok_s);
#   other logs   line by line, each line without the log's time stamp ("0.00.009.685 W "), after dropping the lines
#                that differ by build or by time: the build line (commit), load and run times, and the llama.cpp log
#                lines of the loader and the context (the same calls in both, and not the tools' output);
#   test-hop-cast's TSV, line by line.
# Writes tools.tsv (item, kind, base, head, verdict, detail) and prints a summary; exit 1 if anything differs.
# Usage: python3 -I compare.py <base out dir> <head out dir> <tools.tsv>   (rc.tsv in the dirs is optional)
import re
import sys
from pathlib import Path

TIMING = {"t_gen", "tok_s", "t_race", "t_first", "host_submit_us", "host_wait_us", "host_rest_us", "t_all",
          "total_tok_s"}
# lines that are not the tools' output, or that change with the build or the clock
STAMP = re.compile(r"^\d+\.\d+\.\d+\.\d+ [A-Z] +")
DROP = re.compile(r"^(build: |main: build = |system_info|llama_|llm_|load|print_info|gguf_|common_|sched_|graph_|"
                  r"ggml_|clip_|srv |slot |\.+$)|( ms | ms$|ms per token|tokens per second| t/s|time =|seconds)")


def rc_table(d: Path) -> dict:
    out = {}
    if not (d / "rc.tsv").exists():
        return out
    for line in (d / "rc.tsv").read_text().splitlines():
        if line.strip():
            k, v = line.split("\t")
            out[k] = v
    return out


def pipespec(path: Path) -> dict:
    jobs = {}
    if not path.exists():
        return jobs
    for line in path.read_text(errors="replace").splitlines():
        kind, _, rest = line.partition(" ")
        kv = dict(f.split("=", 1) for f in rest.split() if "=" in f)
        jobs[(kind, kv.get("name", "?"))] = {k: v for k, v in kv.items() if k not in TIMING}
    return jobs


def kept(path: Path) -> list:
    if not path.exists():
        return []
    lines = (STAMP.sub("", ln).rstrip() for ln in path.read_text(errors="replace").splitlines())
    return [ln for ln in lines if ln.strip() and not DROP.search(ln)]


def main() -> int:
    base, head, tsv = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    rows = []
    rb, rh = rc_table(base), rc_table(head)
    for k in sorted(set(rb) | set(rh), key=lambda x: list(rb).index(x) if x in rb else 999):
        b, h = rb.get(k, "-"), rh.get(k, "-")
        rows.append((k, "exit code", b, h, "same" if b == h else "DIFFERS", ""))

    for d in sorted(p.name for p in base.iterdir() if p.is_dir() and p.name.startswith("ps-")):
        jb, jh = pipespec(base / d / "results.txt"), pipespec(head / d / "results.txt")
        for key in sorted(set(jb) | set(jh), key=lambda x: (list(jb).index(x) if x in jb else 999)):
            b, h = jb.get(key), jh.get(key)
            if b is None or h is None:
                rows.append((f"{d}/{key[1]}", key[0], "present" if b else "missing", "present" if h else "missing",
                             "DIFFERS", ""))
                continue
            diff = sorted(k for k in set(b) | set(h) if b.get(k) != h.get(k))
            shown = [k for k in ("n_gen", "tokens", "logits", "state", "probe", "n_conv", "total_tokens") if k in h]
            detail = (" ".join(f"{k} {h[k]}" for k in shown) if not diff
                      else "differ: " + ", ".join(f"{k} {b.get(k)}/{h.get(k)}" for k in diff))
            rows.append((f"{d}/{key[1]}", key[0], f"{len(b)} fields", f"{len(h)} fields",
                         "same" if not diff else "DIFFERS", detail))

    for p in sorted(base.glob("*.log")):
        if p.name.startswith("ps-"):
            continue  # check_identity's and check_streams' tables: their PIPESPEC lines are compared above
        lb, lh = kept(p), kept(head / p.name)
        n = sum(1 for a, b in zip(lb, lh) if a != b) + abs(len(lb) - len(lh))
        first = next((f"{a!r} / {b!r}" for a, b in zip(lb, lh) if a != b), "")
        last = lh[-1][:150] if lh else ""
        rows.append((p.stem, "log", f"{len(lb)} lines", f"{len(lh)} lines", "same" if n == 0 else "DIFFERS",
                     last if n == 0 else f"{n} lines differ; first: {first[:300]}"))

    for p in sorted(base.glob("*.tsv")):
        if p.name == "rc.tsv":
            continue
        a, b = p.read_text().splitlines(), (head / p.name).read_text().splitlines() if (head / p.name).exists() else []
        rows.append((p.stem, "tsv", f"{len(a)} lines", f"{len(b)} lines", "same" if a == b else "DIFFERS", ""))

    with tsv.open("w") as f:
        f.write("item\tkind\tbase\thead\tverdict\tdetail\n")
        for r in rows:
            f.write("\t".join(str(x).replace("\t", " ") for x in r) + "\n")
    bad = [r for r in rows if r[4] != "same"]
    print(f"{len(rows)} comparisons, {len(rows) - len(bad)} same, {len(bad)} differ")
    for r in bad:
        print("  " + " | ".join(str(x) for x in r))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
