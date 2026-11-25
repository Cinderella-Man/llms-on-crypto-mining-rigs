#!/usr/bin/env python3
"""C1038 (Q1771): the tables of this folder, from the new sets' files, R153's manifests and C912's unrecorded cases.

Usage: python3 -I tables.py <in> <out> <unrecorded.tsv>
  <in>: a folder holding, per set S in v4kern and ro8kt: S/manifest.tsv, S/runs.tsv, S/record.txt (the new set),
        r153-S.tsv (R153's manifest of S), replay-S/summary.tsv, replay-S/filters.tsv and replay-S/filter-<n>.log
        (cpu_replay.sh's output), and files.md5 (record_sets.sh's list)
  <out>: the folder for sets.tsv, new-cases.tsv, vs-r153.tsv, cpu-replay.tsv
  <unrecorded.tsv>: docs/data/2026-11-24_c912_p106_check/unrecorded.tsv (the cases C912's check ran unjudged)
A manifest line: id, case, input bytes, input md5, reference bytes, reference md5.
"""
import csv
import os
import sys

SETS = ["v4kern", "ro8kt"]


def read_tsv(path, header=False):
    with open(path, newline="") as f:
        rows = list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    return rows[1:] if header else rows


def manifest(path):
    return {r[0]: r for r in read_tsv(path) if r}


def write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def main():
    src, out, unrec_path = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    unrec = {r[0]: r[1] for r in read_tsv(unrec_path, header=True)}
    files = {}
    for r in read_tsv(os.path.join(src, "files.md5")):
        md5, name = r[0].split("  ", 1)
        files[name] = md5

    sets, vs, newc, rep = [], [], {}, []
    for s in SETS:
        new = manifest(os.path.join(src, s, "manifest.tsv"))
        old = manifest(os.path.join(src, f"r153-{s}.tsv"))
        tbo = ""
        with open(os.path.join(src, s, "record.txt")) as f:
            for line in f:
                if line.startswith("test-backend-ops md5:"):
                    tbo = line.split(":", 1)[1].strip()
        runs = [r for r in read_tsv(os.path.join(src, s, "runs.tsv")) if r]
        new_names = {r[1]: True for r in new.values()}
        only_new = sorted(set(new) - set(old))
        only_old = sorted(set(old) - set(new))
        both = sorted(set(new) & set(old))
        same_name = sum(new[i][1] == old[i][1] for i in both)
        size_in = sum(new[i][2] == old[i][2] for i in both)
        size_ref = sum(new[i][4] == old[i][4] for i in both)
        same_in = sum(new[i][3] == old[i][3] and new[i][2] == old[i][2] for i in both)
        same_ref = sum(new[i][5] == old[i][5] and new[i][4] == old[i][4] for i in both)
        nbytes = sum(int(r[2]) + int(r[4]) for r in new.values())
        sets.append([s, len(new), len(old), len(only_new), len(only_old), nbytes, len(runs), tbo,
                     sum(1 for k in files if k.startswith(s + "/")),
                     sum(i in new for i in unrec), len(unrec)])
        vs.append([s, len(both), same_name, size_in, size_ref, same_in, same_ref])
        for i in only_new:
            r = new[i]
            e = newc.setdefault(i, [i, r[1], r[2], r[3], r[4], r[5], [], "yes" if i in unrec else "no"])
            e[6].append(s)
        # the CPU replay: per filter of the set's run list
        filt = {r[0]: r for r in read_tsv(os.path.join(src, f"replay-{s}", "filters.tsv")) if r}
        for r in read_tsv(os.path.join(src, f"replay-{s}", "summary.tsv"), header=True):
            n, cases, ok, fail, norec, summ, rc = r
            f = filt[n]
            # cases the CPU backend declines as the tested backend (test-backend-ops prints "not supported [CPU]"):
            # skipped before the record is looked up, so neither replayed nor counted as NOREC
            with open(os.path.join(src, f"replay-{s}", f"filter-{n}.log"), errors="replace") as fl:
                declined = sorted({line.split("): not supported [CPU]")[0].strip() + ")"
                                   for line in fl if "): not supported [CPU]" in line})
            held = sum(new_names.get(c, False) for c in declined)
            rep.append([s, n, f[1], f[2], len(f[3].split()), cases, ok, fail, norec, len(declined), held, summ, rc])

    write(os.path.join(out, "sets.tsv"),
          ["set", "cases", "cases_r153", "new_cases", "cases_r153_only", "bytes", "runs", "test_backend_ops_md5",
           "files", "c912_unrecorded_now_recorded", "c912_unrecorded"], sets)
    write(os.path.join(out, "vs-r153.tsv"),
          ["set", "cases_in_both", "same_case_string", "same_input_bytes", "same_reference_bytes", "same_input_md5",
           "same_reference_md5"], vs)
    write(os.path.join(out, "new-cases.tsv"),
          ["id", "case", "input_bytes", "input_md5", "reference_bytes", "reference_md5", "sets", "c912_unrecorded"],
          [e[:6] + [" ".join(e[6]), e[7]] for e in sorted(newc.values(), key=lambda e: e[1])])
    write(os.path.join(out, "cpu-replay.tsv"),
          ["set", "filter", "ops", "params", "runs_using_it", "cases", "ok", "fail", "norec", "declined_by_cpu",
           "declined_but_recorded", "replay_summary", "exit"], rep)


if __name__ == "__main__":
    main()
