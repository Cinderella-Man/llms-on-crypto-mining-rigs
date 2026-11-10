#!/usr/bin/env python3
"""Tables of a scripts/replay_checks.sh run: per run, per case, and the sparse decode attention's cases by name.

Reads the replay's output folders (console.txt, results.tsv, <run>.log, <run>.tsv; OUT may name several, separated
by spaces) and the recorded suite's index files (runs.tsv, manifest.tsv), and writes into DST:
  runs.tsv          one line per run: its switches, what passing means, the recorded cases its filter selects, the
                    cases replayed, passed and failed, the selected records not replayed (the card declines the case,
                    or the run stopped before it), the cases the binary selects that the record lacks, the exit
                    status, replay_checks.sh's verdict, the run's seconds, and what the log says when the run stopped
                    or compute-sanitizer refused the card
  cases.tsv.gz      one line per replayed case of every run: run, record id, case, OK or FAIL, each compared node's error
  sparse_cases.tsv  the FLASH_ATTN_EXT cases by name, per sparse run OK or FAIL and the compared node's error
  sparse_cases.md   the same cases as a table of their parameters, with the runs whose error differs from the
                    switch-off run's marked (another kernel computed the case)
  unmatched.tsv     recorded cases that no run replayed

Usage: OUT="<replay output> [<a second one>]" SET=<folder with runs.tsv and manifest.tsv> DST=<dir> python3 tables.py
"""
import csv
import datetime
import gzip
import os
import re
import sys

OUTS = os.environ["OUT"].split()
SET = os.environ["SET"]
DST = os.environ["DST"]
os.makedirs(DST, exist_ok=True)

# manifest.tsv: id, case "OP(vars)", input bytes, input md5, reference bytes, reference md5
manifest = {}
with open(os.path.join(SET, "manifest.tsv")) as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        op, _, rest = p[1].partition("(")
        manifest[p[0]] = (op, rest[:-1], p[1])


def selected(ops, params):
    """The recorded cases a run's -o and -p select: -o holds exact op names, -p is a regex searched in the vars."""
    names = set(ops.split(","))
    rx = re.compile(params)
    return {i for i, (op, vs, _) in manifest.items() if op in names and rx.search(vs)}


# runs.tsv of the suite: run, switches, -o, -p, expect, wrap
suite = []
with open(os.path.join(SET, "runs.tsv")) as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 6 and p[0]:
            suite.append(p)

# each output folder: its runs in the order they ran, and each run's seconds from the logs' modification times
results = {}
where = {}
seconds = {}
for out in OUTS:
    with open(os.path.join(out, "console.txt")) as f:
        t = datetime.datetime.fromisoformat(f.readline().split()[1]).timestamp()
    with open(os.path.join(out, "results.tsv")) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            run = row["run"]
            results[run] = row
            where[run] = out
            end = os.path.getmtime(os.path.join(out, run + ".log"))
            seconds[run] = end - t
            t = end

replayed_ids = set()
sparse_runs = [p[0] for p in suite if p[2] == "FLASH_ATTN_EXT"]
sparse = {}
n_cases = 0
with open(os.path.join(DST, "runs.tsv"), "w") as fr, gzip.open(os.path.join(DST, "cases.tsv.gz"), "wt") as fc:
    fr.write("run\tswitches\texpect\twrap\trecorded_selected\treplayed\tpassed\tfailed\tselected_not_replayed\t"
             "not_recorded\texit\tverdict\tseconds\tlog\n")
    fc.write("run\tid\tcase\tverdict\terrors\n")
    for run, envs, ops, params, expect, wrap in suite:
        want = selected(ops, params)
        r = results.get(run)
        if r is None:
            fr.write(f"{run}\t{envs}\t{expect}\t{wrap}\t{len(want)}\t\t\t\t\t\t\tnot run\t\t\n")
            continue
        ok = fail = 0
        got = set()
        path = os.path.join(where[run], run + ".tsv")
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    p = line.rstrip("\n").split("\t")
                    if len(p) < 3:
                        continue
                    got.add(p[0])
                    ok += p[2] == "OK"
                    fail += p[2] != "OK"
                    fc.write(f"{run}\t{p[0]}\t{p[1]}\t{p[2]}\t{p[3] if len(p) > 3 else ''}\n")
                    n_cases += 1
                    if run in sparse_runs:
                        sparse.setdefault(p[1], {})[run] = (p[2], p[3] if len(p) > 3 else "")
        assert got <= want, f"{run}: replayed cases outside the recorded selection"
        replayed_ids |= got
        missing = ""
        notes = []
        with open(os.path.join(where[run], run + ".log"), errors="replace") as f:
            for line in f:
                m = re.search(r"(\d+) cases replayed, (\d+) selected by the filter have no record", line)
                if m:
                    missing = m.group(2)
                if line.startswith("CUDA error: "):
                    notes.append(line.strip())
                m = re.match(r"=+ (Error: .*?\.) ", line)
                if m:
                    notes.append("compute-sanitizer: " + m.group(1))
        fr.write(f"{run}\t{envs}\t{expect}\t{wrap}\t{len(want)}\t{len(got)}\t{ok}\t{fail}\t{len(want - got)}\t"
                 f"{missing}\t{r['exit']}\t{r['verdict']}\t{seconds[run]:.0f}\t{'; '.join(notes)}\n")

with open(os.path.join(DST, "sparse_cases.tsv"), "w") as f:
    f.write("case\t" + "\t".join(sparse_runs) + "\t" + "\t".join("err_" + r for r in sparse_runs) + "\n")
    for case in sorted(sparse):
        f.write(case + "\t" + "\t".join(sparse[case].get(r, ("-", ""))[0] for r in sparse_runs) + "\t"
                + "\t".join(sparse[case].get(r, ("-", ""))[1] for r in sparse_runs) + "\n")

# sparse_cases.md: the same cases as a readable table. A case's first compared node is the attention's output; where
# its error with the switch on differs from the error with it off, another kernel computed it (the sparse one).
if sparse:
    def err(case, run):
        return sparse[case].get(run, ("-", ""))[1].split(",")[0]

    def params(case):
        return dict(re.findall(r"(\w+)=(\[[^\]]*\]|[^,)]*)", case))

    off = next((r for r in sparse_runs if r.endswith("sparse0")), None)
    with open(os.path.join(DST, "sparse_cases.md"), "w") as f:
        f.write("| # | head K/V | nh | nr23 | kv | nb | K/V type | n_kv_max | other | "
                + " | ".join(r.replace("qsa-", "") for r in sparse_runs)
                + " | error, sparse1 | error, sparse2 | error, off | sparse kernel at 1 | at 2 |\n")
        f.write("|" + "---|" * (14 + len(sparse_runs)) + "\n")
        for n, case in enumerate(sorted(sparse), 1):
            p = params(case)
            other = [k + "=" + p[k] for k, plain in (("sinks", "0"), ("max_bias", "0.000000"),
                                                     ("permute", "[0,1,2,3]"), ("v_is_view_of_k", "0"))
                     if p.get(k, plain) != plain]
            types = p["type_K"] if p["type_K"] == p["type_V"] else p["type_K"] + "/" + p["type_V"]
            e = {r: err(case, r) for r in sparse_runs}
            took = ["yes" if off and e.get(r) != e.get(off) else "no"
                    for r in sparse_runs if r.endswith(("sparse1", "sparse2"))]
            f.write(f"| {n} | {p['hsk']}/{p['hsv']} | {p['nh']} | {p['nr23']} | {p['kv']} | {p['nb']} | {types} | "
                    f"{p['n_kv_max']} | {', '.join(other)} | "
                    + " | ".join(sparse[case].get(r, ("-", ""))[0] for r in sparse_runs)
                    + f" | {e['qsa-sparse1']} | {e['qsa-sparse2']} | {e[off]} | "
                    + " | ".join(took) + " |\n")

with open(os.path.join(DST, "unmatched.tsv"), "w") as f:
    f.write("id\tcase\n")
    for i in sorted(set(manifest) - replayed_ids):
        f.write(f"{i}\t{manifest[i][2]}\n")

print(f"{len(suite)} runs in the suite, {len(results)} run, {n_cases} case lines, "
      f"{len(manifest)} records, {len(set(manifest) - replayed_ids)} never replayed", file=sys.stderr)
