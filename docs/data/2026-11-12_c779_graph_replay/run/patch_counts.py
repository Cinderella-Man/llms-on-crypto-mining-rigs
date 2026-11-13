"""patch_counts.py: the GGML_CUDA_GRAPHS_PATCH counts lines of a llama-server log as a TSV (one line per 64 decisions)."""
import re, sys
pat = re.compile(r"^(\S+) W GGML_CUDA_GRAPHS_PATCH: (\d+) of (\d+) captures patched \((\d+) groups, (\d+) ggml nodes, (\d+) CUDA nodes set\); "
                 r"captured in full: not_ready (\d+) structure (\d+) share (\d+) skip (\d+) pool (\d+) fuse (\d+) nodes (\d+) api (\d+)")
print("log_time\tdecisions\tpatched\tgroups\tggml_nodes\tcuda_nodes_set\tnot_ready\tstructure\tshare\tskip\tpool\tfuse\tnodes\tapi")
for line in open(sys.argv[1]):
    m = pat.match(line)
    if m:
        g = m.groups()
        print("\t".join([g[0], g[2], g[1]] + list(g[3:])))
