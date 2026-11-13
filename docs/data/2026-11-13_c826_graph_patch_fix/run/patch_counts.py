"""patch_counts.py (C826): the GGML_CUDA_GRAPHS_PATCH lines of a llama-server log as TSV.

Reads C779's counts line (one per 64 decisions) and C826's, which ends with the memset and memcpy nodes kept; C779's
logs leave that column empty. Modes:
  patch_counts.py LOG               the counts lines
  patch_counts.py --refused LOG     every refused update by call, node type, ggml op and CUDA error: its count in the
                                    last "refused updates" line, and the node and parameters its first "CUDA refused"
                                    line names; then a check line: the last counts line's api count against the sum
  patch_counts.py --structure LOG   the "structure changed" lines (the first 16 of the process)
"""
import re
import sys

COUNTS = re.compile(r"^(\S+) W GGML_CUDA_GRAPHS_PATCH: (\d+) of (\d+) captures patched \((\d+) groups, (\d+) ggml nodes, "
                    r"(\d+) CUDA nodes set\); captured in full: not_ready (\d+) structure (\d+) share (\d+) skip (\d+) "
                    r"pool (\d+) fuse (\d+) nodes (\d+) api (\d+)(?:; memset and memcpy nodes kept (\d+))?")
FIRST = re.compile(r"^(\S+) W GGML_CUDA_GRAPHS_PATCH: CUDA refused (\S+) (\S+) (\S+) (\S+): (.*)$")
SUMMARY = re.compile(r"^(\S+) W GGML_CUDA_GRAPHS_PATCH: refused updates: (.*)$")
STRUCTURE = re.compile(r"^(\S+) W GGML_CUDA_GRAPHS_PATCH: structure changed at node (\d+) of (\d+) \((.*)\): (\S+) of (.*)$")


def lines(path):
    with open(path, errors="replace") as f:
        for line in f:
            yield line.rstrip("\n")


def counts(path):
    print("log_time\tdecisions\tpatched\tgroups\tggml_nodes\tcuda_nodes_set\tnot_ready\tstructure\tshare\tskip\tpool\tfuse"
          "\tnodes\tapi\tkept")
    for line in lines(path):
        m = COUNTS.match(line)
        if m:
            g = list(m.groups())
            g[-1] = g[-1] or ""
            print("\t".join([g[0], g[2], g[1]] + g[3:]))


def refused(path):
    first, last, api = {}, {}, None
    for line in lines(path):
        m = FIRST.match(line)
        if m:
            key = m.group(2, 3, 4, 5)
            first.setdefault(key, (m.group(1), m.group(6)))
            continue
        m = SUMMARY.match(line)
        if m:
            last = {}
            for item in m.group(2).split("; "):
                k, n = item.rsplit(" ", 1)
                last[tuple(k.split(" "))] = int(n)
            continue
        m = COUNTS.match(line)
        if m:
            api = int(m.group(14))
    print("call\tnode_type\tggml_op\tcuda_error\tcount\tfirst_logged\tnode_and_parameters")
    for key in sorted(set(first) | set(last)):
        t, rest = first.get(key, ("", ""))
        print("\t".join(list(key) + [str(last.get(key, "")), t, rest]))
    total = sum(last.values())
    verdict = "named" if api is not None and api == total else "NOT all named"
    print(f"# api in the last counts line: {api}; refused updates counted: {total}; {verdict}")


def structure(path):
    print("log_time\tnode\tn_nodes\tname_and_op\tfield\tof")
    for line in lines(path):
        m = STRUCTURE.match(line)
        if m:
            print("\t".join(m.groups()))


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == "--refused":
        refused(args[1])
    elif len(args) == 2 and args[0] == "--structure":
        structure(args[1])
    elif len(args) == 1:
        counts(args[0])
    else:
        sys.exit(__doc__)
