#!/usr/bin/env python3
# C1025 (Q1758): the tables of run.sh's check step.
#   c1025_tables.py <work dir> <out dir>
# Writes list.tsv (the device list per arm: exit code, libnvidia-ml in the loader's trace, the head's probe line),
# load.tsv (one load per arm: up, libnvidia-ml, the probe line, the two greedy replies against the candidate's) and
# cublas.tsv (every cuBLAS call of a load by function and compute type), and prints a summary.
import collections, glob, gzip, json, os, re, sys

W, OUT = sys.argv[1], sys.argv[2]
C = os.path.join(W, "check")
os.makedirs(OUT, exist_ok=True)


def read(p):
    try:
        with open(p, errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def nvml_line(text):
    lines = [l.strip() for l in text.splitlines() if "NVML probe" in l or "GGML_CUDA_NVML_PROBE" in l]
    # drop the log prefix up to the function name, and the probe's times (printed, not results on this machine)
    out = []
    for l in lines:
        l = re.sub(r"^.*?(ggml_cuda_init:)", r"\1", l)
        l = re.sub(r"open and init [0-9.]+ ms, shutdown and close [0-9.]+ ms", "open and init <t> ms, shutdown and close <t> ms", l)
        out.append(l)
    return " | ".join(out) if out else "-"


def nvml_loads(prefix):
    n = 0
    for p in glob.glob(prefix + ".*"):
        n += read(p).count("file=libnvidia-ml")
    return n


def tokens(p):
    try:
        d = json.load(open(p))
    except Exception:
        return None, ""
    return d.get("tokens"), d.get("content", "")


summary = []
with open(os.path.join(OUT, "list.tsv"), "w") as f:
    f.write("arm\trc\tlibnvidia_ml_loads\tprobe_line\n")
    for arm in ("C-unset", "H-unset", "H-0", "H-1", "H-2"):
        rc = read(os.path.join(C, f"list-{arm}.rc")).strip() or "-"
        n = nvml_loads(os.path.join(C, f"ld-list-{arm}"))
        line = nvml_line(read(os.path.join(C, f"list-{arm}.log")))
        f.write(f"{arm}\t{rc}\t{n}\t{line}\n")
        summary.append(f"list {arm}: {rc}, libnvidia-ml loaded {n} time(s); {line}")

ref = {p: tokens(os.path.join(C, "load-C-unset", f"reply-{p}.json")) for p in ("long", "short")}
calls = {}
with open(os.path.join(OUT, "load.tsv"), "w") as f:
    f.write("arm\tup\tlibnvidia_ml_loads\tprobe_line\treply\ttokens\tsame_token_ids_as_candidate\tsame_text\n")
    for arm in ("C-unset", "H-unset", "H-0", "H-1"):
        L = os.path.join(C, f"load-{arm}")
        up = read(os.path.join(L, "load.txt")).strip() or "-"
        n = nvml_loads(os.path.join(L, "ld"))
        line = nvml_line(read(os.path.join(L, "server.log")))
        for p in ("long", "short"):
            t, txt = tokens(os.path.join(L, f"reply-{p}.json"))
            rt, rtxt = ref[p]
            if t is None or rt is None:
                same, first = "missing", "-"
            else:
                same = "yes" if t == rt else "no"
                first = next((i for i, (x, y) in enumerate(zip(t, rt)) if x != y), min(len(t), len(rt)))
            f.write(f"{arm}\t{up}\t{n}\t{line}\t{p}\t{len(t) if t else 0}\t{same}\t{'yes' if txt == rtxt else 'no'}\n")
            summary.append(f"load {arm} {p}: {up}, libnvidia-ml {n}, {len(t) if t else 0} tokens, "
                           f"token ids as the candidate's: {same}" + (f" (first difference at {first})" if same == "no" else ""))
        summary.append(f"load {arm}: {line}")
        # cuBLAS's API log: "I! cuBLAS (...) function <ret> <name>(<args>) called:" then "i!  computeType: ...; val=<X>(<n>)"
        cnt = collections.Counter()
        fn = None
        p = os.path.join(L, "cublas.log.gz")
        opener = gzip.open if os.path.exists(p) else open
        if not os.path.exists(p):
            p = os.path.join(L, "cublas.log")
        try:
            with opener(p, "rt", errors="replace") as g:
                for l in g:
                    m = re.match(r"^I! cuBLAS .*? function .*?(cublas\w+)\(", l)
                    if m:
                        if fn is not None:
                            cnt[(fn, ct)] += 1
                        fn, ct = m.group(1), "-"
                        continue
                    m = re.match(r"^i!\s+computeType: .*?val=(\w+)", l)
                    if m and fn is not None:
                        ct = m.group(1)
                if fn is not None:
                    cnt[(fn, ct)] += 1
        except OSError:
            pass
        calls[arm] = cnt

with open(os.path.join(OUT, "cublas.tsv"), "w") as f:
    keys = sorted(set(k for c in calls.values() for k in c))
    arms = list(calls)
    f.write("function\tcompute_type\t" + "\t".join(arms) + "\n")
    for k in keys:
        f.write(f"{k[0]}\t{k[1]}\t" + "\t".join(str(calls[a].get(k, 0)) for a in arms) + "\n")
gemm = {a: {k: v for k, v in c.items() if k[1] != "-"} for a, c in calls.items()}
for a in calls:
    summary.append(f"cuBLAS {a}: {sum(calls[a].values())} calls; with a compute type: "
                   + (", ".join(f"{k[0]} {k[1]} {v}" for k, v in sorted(gemm[a].items())) or "none")
                   + ("; the same counts as the candidate's" if a != "C-unset" and calls[a] == calls.get("C-unset") else ""))
print("\n".join(summary))
