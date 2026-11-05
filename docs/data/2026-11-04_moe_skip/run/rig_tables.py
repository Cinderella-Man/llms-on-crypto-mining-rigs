#!/usr/bin/env python3
"""rig_tables.py: the tables of rig_tk.sh's run. T(k) per arm from requests.tsv (rows "tk-<arm>"), each arm's change
against the switch off (the mean of the arms off and off2, two loads at the run's start and end, whose difference is
the spread between loads), and, with --parts (kpass_parts.py's parts.tsv of the prof traces), the parts at k = 1 and 5.

  python3 -I rig_tables.py --requests OUT/requests.tsv [--parts <dir>/parts.tsv] --out <dir>

Writes <out>/tk.tsv (k, T(k) per arm, change per arm against off) and <out>/parts_k15.tsv, and prints the Markdown
tables. Standard library only."""
import argparse, csv, os

ap = argparse.ArgumentParser()
ap.add_argument("--requests", required=True); ap.add_argument("--parts", default=""); ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

t = {}  # arm -> k -> ms per micro-batch
for r in csv.DictReader(open(a.requests), delimiter="\t"):
    if r["arm"].startswith("tk-"):
        t.setdefault(r["arm"][3:], {})[int(r["k"])] = float(r["ms_per_micro_batch"])
arms = [x for x in ("off", "p090r", "p080r", "p090rr", "off2") if x in t]
ks = sorted(set(k for x in arms for k in t[x]))
base = {k: sum(t[x][k] for x in ("off", "off2") if x in t and k in t[x]) / sum(1 for x in ("off", "off2") if x in t and k in t[x]) for k in ks}
cmp_arms = [x for x in ("p090r", "p080r", "p090rr") if x in t]

with open(os.path.join(a.out, "tk.tsv"), "w") as f:
    f.write("k\t" + "\t".join(f"T_{x}" for x in arms) + "\toff2_minus_off\t" + "\t".join(f"d_{x}" for x in cmp_arms)
            + "\td_p090rr_minus_p090r\n")
    for k in ks:
        row = [str(k)] + [f"{t[x].get(k, float('nan')):.2f}" for x in arms]
        row.append(f"{t['off2'][k] - t['off'][k]:.2f}" if "off2" in t and "off" in t else "")
        row += [f"{t[x][k] - base[k]:.2f}" for x in cmp_arms]
        row.append(f"{t['p090rr'][k] - t['p090r'][k]:.2f}" if "p090rr" in t and "p090r" in t else "")
        f.write("\t".join(row) + "\n")

print("| k | " + " | ".join(f"`{x}`" for x in arms) + " | `off2` − `off` | " + " | ".join(f"`{x}` − off" for x in cmp_arms)
      + " | `p090rr` − `p090r` |")
print("|---:" * (1 + len(arms) + 1 + len(cmp_arms) + 1) + "|")
for k in ks:
    cells = [f"{t[x][k]:.1f}" for x in arms]
    cells.append(f"{t['off2'][k] - t['off'][k]:+.1f}" if "off2" in t else "")
    cells += [f"{t[x][k] - base[k]:+.1f} ({100 * (t[x][k] - base[k]) / base[k]:+.1f} %)" for x in cmp_arms]
    cells.append(f"{t['p090rr'][k] - t['p090r'][k]:+.1f}" if "p090rr" in t and "p090r" in t else "")
    print(f"| {k} | " + " | ".join(cells) + " |")

if a.parts:
    p = {}  # (tag, k) -> part -> ms
    order = []
    for r in csv.DictReader(open(a.parts), delimiter="\t"):
        p.setdefault((r["trace"], int(r["k"])), {})[r["part"]] = float(r["ms_per_pass"])
        if r["part"] not in order:
            order.append(r["part"])
    keys = sorted(p, key=lambda x: (x[1], x[0] != "off", x[0]))
    with open(os.path.join(a.out, "parts_k15.tsv"), "w") as f:
        f.write("part\t" + "\t".join(f"{tag}_k{k}" for tag, k in keys) + "\n")
        for part in order:
            f.write(part + "\t" + "\t".join(f"{p[key].get(part, float('nan')):.2f}" for key in keys) + "\n")
    print()
    print("| Part | " + " | ".join(f"`{tag}`, k = {k}" for tag, k in keys) + " |")
    print("|---" + "|---:" * len(keys) + "|")
    for part in order:
        print(f"| {part} | " + " | ".join(f"{p[key].get(part, float('nan')):.2f}" for key in keys) + " |")
