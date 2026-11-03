#!/usr/bin/env python3
# fit_table.py PROJ.csv PARTS.csv LOG...: scripts/fit_probe.sh logs as tables. PROJ.csv has one row per configuration
# (file, engine, context, micro-batch, cache type, -ts, -ot moves, MTP draft): llama.cpp's projected MiB per card (the
# main model plus, with a draft, the draft's weights, context and compute), the smallest free MiB and its card, the
# total on the cards, the host memory (mapped: the weights that stay in host memory, read through the page cache, here
# the n-gram table and token embeddings; allocated: context and compute buffers in RAM), and whether every card keeps
# 150 MiB. PARTS.csv has the memory breakdown behind it: per configuration, model (main or draft) and card, the
# weights, context (KV cache and recurrent state) and compute MiB. Standard library only.
import csv, re, sys

KEYS = ("config", "model_file", "engine", "ctx", "ub", "kv", "ts", "ot", "mtp_draft", "draft_device")

ENGINES = {"d77256999": "stock d77256999", "37e717ff6": "v2.0 = v3 base 37e717ff6"}
BREAK = re.compile(r"common_memory_breakdown_print: \|\s+- (CUDA\d+|Host)\b[^|]*\|\s*(.*?)\s*\|$")
GPU = re.compile(r"(\d+) = (\d+) \+ \(\s*(\d+) =\s*(\d+) \+\s*(\d+) \+\s*(\d+)\)")
HOST = re.compile(r"(\d+) =\s*(\d+) \+\s*(\d+) \+\s*(\d+)")
PROJ = re.compile(r"common_params_fit_impl:\s+- CUDA(\d+) \([^)]*\):\s+(\d+) total,\s+(-?\d+) used,\s+(-?\d+) free")


def opt(args, name):
    m = re.search(rf"(?:^| ){re.escape(name)} (\S+)", args)
    return m.group(1) if m else ""


def main():
    proj_out, parts_out, logs = sys.argv[1], sys.argv[2], sys.argv[3:]
    rows, parts = [], []
    cur = None
    model_file = ""
    for path in logs:
        for line in open(path, encoding="utf-8", errors="replace"):
            line = line.rstrip("\n")
            if line.startswith("# fit_probe "):
                model_file = re.sub(r"-0000\d-of-0000\d\.gguf$", "", line.split()[2])
                continue
            m = re.match(r"== (\S+): (\S+) -c (\d+) ?(.*)$", line)
            if m:
                label, build, ctx, args = m.groups()
                engine = next((v for k, v in ENGINES.items() if k in build), build)
                cur = {"config": label, "model_file": model_file, "engine": engine, "ctx": int(ctx),
                       "ub": opt(args, "-ub"), "kv": opt(args, "-ctk") or "f16", "ts": opt(args, "-ts") or "default",
                       "ot": opt(args, "-ot").strip("'"),
                       "mtp_draft": ("n_max " + opt(args, "--spec-draft-n-max")) if "-md " in args else "",
                       "draft_device": opt(args, "-devd"), "cards": {}, "free": {}, "host_model": 0, "host_compute": 0,
                       "blocks": 0}
                rows.append(cur)
                continue
            if cur is None:
                continue
            b = BREAK.search(line)
            if b:
                dev, rest = b.groups()
                if dev == "CUDA0":
                    cur["blocks"] += 1
                part = "main" if cur["blocks"] == 1 else "draft"
                if dev == "Host":
                    h = HOST.search(rest)
                    if h:
                        cur["host_model"] += int(h.group(2))
                        cur["host_compute"] += int(h.group(3)) + int(h.group(4))
                        parts.append([cur["config"], part, "host", h.group(2), h.group(3), h.group(4)])
                else:
                    g = GPU.search(rest)
                    if g:
                        parts.append([cur["config"], part, dev[4:], g.group(4), g.group(5), g.group(6)])
                continue
            p = PROJ.search(line)
            if p:
                card, _total, used, free = map(int, p.groups())
                cur["cards"][card] = used
                cur["free"][card] = free
    with open(proj_out, "w", newline="") as f:
        w = csv.writer(f)
        n = max((len(r["cards"]) for r in rows), default=0)
        w.writerow(list(KEYS) + [f"card{i}_mib" for i in range(n)]
                   + ["min_free_mib", "min_free_card", "device_total_mib", "host_mapped_mib", "host_alloc_mib",
                      "fits_150mib"])
        for r in rows:
            if not r["free"]:
                w.writerow([r[k] for k in KEYS] + [""] * (n + 5) + ["no projection"])
                continue
            card = min(r["free"], key=r["free"].get)
            w.writerow([r[k] for k in KEYS] + [r["cards"].get(i, "") for i in range(n)]
                       + [r["free"][card], card, sum(r["cards"].values()), r["host_model"], r["host_compute"],
                          "yes" if r["free"][card] >= 150 else "no"])
    with open(parts_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "part", "card", "model_mib", "context_mib", "compute_mib"])
        w.writerows(parts)


if __name__ == "__main__":
    main()
