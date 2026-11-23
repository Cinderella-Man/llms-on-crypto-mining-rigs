#!/usr/bin/env python3
# C957 (2026-11-23): the published tables from run.sh's outputs (WORK/tools/<run>/, WORK/compare/).
#   checks.tsv    one row per compared pair of runs: what the pair shows, compare.py's counts, and what differs
#   rc.tsv        every tool's and test's exit code in every run (rows: the item, columns: the runs)
#   tests.tsv     every tool's and test's verdict line in every run (the last line of its log, as compare.py shows it)
#   pipespec.tsv  every pipespec job of every run: tokens generated, the tokens' and logits' hashes, and whether its
#                 tokens, logits, KV state and probe equal its reference's (the plain run in check_identity.sh; the
#                 conversation's plain reference ref-cN in check_streams.sh): the checks' own verdict, run by run
#   pairs/        compare.py's table of each pair, as written
# No timing column is read or written.
# Usage: python3 -I summary.py <WORK> <out dir>
import re
import shutil
import sys
from pathlib import Path

RUNS = ["f32-stock-base", "f32-stock-head", "f32-stock-base2", "f32-p106-base", "f32-p106-head",
        "q8-p106-base", "q8-p106-head", "q8-stock-base"]
PAIRS = [
    ("f32-stock-base", "f32-stock-head", "the check: base against branch, F32 models, every switch at its default"),
    ("f32-p106-base", "f32-p106-head", "the check: base against branch, F32 models, the fork's Pascal kernels forced"),
    ("q8-p106-base", "q8-p106-head", "the check: base against branch, Q8_0 models, the fork's Pascal kernels forced"),
    ("f32-stock-base", "f32-stock-base2", "the runs repeat: the base twice"),
    ("f32-stock-base", "f32-p106-base", "the setting's reach on F32 weights (the p106 run has no pool tests)"),
    ("q8-stock-base", "q8-p106-base", "the setting's reach on Q8_0 weights"),
]
STAMP = re.compile(r"^\d+\.\d+\.\d+\.\d+ [A-Z] +")
TIMING = {"t_gen", "tok_s", "t_race", "t_first", "host_submit_us", "host_wait_us", "host_rest_us", "t_all",
          "total_tok_s"}


def rc(run_dir: Path) -> dict:
    p = run_dir / "rc.tsv"
    if not p.exists():
        return {}
    return dict(line.split("\t") for line in p.read_text().splitlines() if line.strip())


def last_line(p: Path) -> str:
    lines = [STAMP.sub("", ln).rstrip() for ln in p.read_text(errors="replace").splitlines()]
    keep = [ln for ln in lines if re.match(r"^(PIPE-RECURRENT-CHECK|SCREEN CHECK|\d+ checks, )", ln)]
    return keep[-1] if keep else "-"


def jobs(p: Path) -> list:
    out = []
    if not p.exists():
        return out
    for line in p.read_text(errors="replace").splitlines():
        kind, _, rest = line.partition(" ")
        if not kind.startswith("PIPESPEC"):
            continue
        kv = dict(f.split("=", 1) for f in rest.split() if "=" in f)
        out.append((kind, {k: v for k, v in kv.items() if k not in TIMING}))
    return out


def main() -> int:
    work, out = Path(sys.argv[1]), Path(sys.argv[2])
    tools, cmp_dir = work / "tools", work / "compare"
    runs = [r for r in RUNS if (tools / r).is_dir()]
    (out / "pairs").mkdir(parents=True, exist_ok=True)

    with (out / "checks.tsv").open("w") as f:
        f.write("run a\trun b\twhat it shows\tcomparisons\tsame\tdiffer\twhat differs\n")
        for a, b, what in PAIRS:
            t = cmp_dir / f"{a}__{b}.tsv"
            if not t.exists():
                continue
            shutil.copy(t, out / "pairs" / t.name)
            rows = [ln.split("\t") for ln in t.read_text().splitlines()[1:] if ln.strip()]
            bad = [r for r in rows if r[4] != "same"]
            kinds = {}
            for r in bad:
                if {"-", "missing", "0 lines"} & {r[2], r[3]}:
                    key = "absent in one run"
                elif r[5].startswith("differ: "):
                    key = "differ: " + ", ".join(re.findall(r"(\w+) [^,]*?/", r[5][8:]))
                else:
                    key = r[1]
                kinds[key] = kinds.get(key, 0) + 1
            f.write(f"{a}\t{b}\t{what}\t{len(rows)}\t{len(rows) - len(bad)}\t{len(bad)}\t"
                    + ("; ".join(f"{n} {k}" for k, n in kinds.items()) or "-") + "\n")

    rcs = {r: rc(tools / r) for r in runs}
    items = []
    for r in runs:
        for k in rcs[r]:
            if k not in items:
                items.append(k)
    with (out / "rc.tsv").open("w") as f:
        f.write("item\t" + "\t".join(runs) + "\n")
        for k in items:
            f.write(k + "\t" + "\t".join(rcs[r].get(k, "-") for r in runs) + "\n")

    logs = []
    for r in runs:
        for p in sorted((tools / r).glob("*.log")):
            if not p.name.startswith("ps-") and p.stem not in logs:
                logs.append(p.stem)
    with (out / "tests.tsv").open("w") as f:
        f.write("log\t" + "\t".join(runs) + "\n")
        for name in logs:
            f.write(name + "\t" + "\t".join(
                last_line(tools / r / f"{name}.log") if (tools / r / f"{name}.log").exists() else "-" for r in runs)
                + "\n")

    with (out / "pipespec.tsv").open("w") as f:
        f.write("run\tcheck\tline\tjob\treference\tn_gen\ttokens\tlogits\ttokens as ref\tlogits as ref\tstate as ref"
                "\tprobe as ref\n")
        for r in runs:
            for d in sorted(p for p in (tools / r).iterdir() if p.is_dir() and p.name.startswith("ps-")):
                js = jobs(d / "results.txt")
                by_name = {kv.get("name"): kv for _, kv in js}
                for kind, kv in js:
                    name = kv.get("name", "?")
                    if kind == "PIPESPEC_STREAMS":
                        ref = None
                    elif d.name == "ps-streams":
                        m = re.search(r"-c(\d+)$", name)
                        ref = None if name.startswith("ref-") or not m else f"ref-c{m.group(1)}"
                    else:
                        ref = None if name == "plain" else "plain"
                    rk = by_name.get(ref) if ref else None

                    def same(field):
                        if rk is None or field not in kv:
                            return "-"
                        return "same" if kv[field] == rk.get(field) else "DIFF"
                    f.write("\t".join([r, d.name, kind, name, ref or "-", kv.get("n_gen", kv.get("total_tokens", "-")),
                                       kv.get("tokens", "-"), kv.get("logits", "-"), same("tokens"), same("logits"),
                                       same("state"), same("probe")]) + "\n")
    print(f"{len(runs)} runs; tables in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
