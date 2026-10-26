"""Summary of a run's sessions: per session and in total, turns, generated tokens, how the session ended, whether its
task's check passed, tool calls by tool, failed and unknown calls, the share of the sequence the model generated, and
the time spent in tools. Token counts only: times on a shared machine are not speed results.

usage: python report.py SESSIONS_DIR [SESSIONS_DIR ...] [--md]
"""
import argparse
import collections
import json
import os
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--md", action="store_true", help="print the per-session table as Markdown")
    a = ap.parse_args()
    rows, calls, tot = [], collections.Counter(), collections.Counter()
    for d in a.dirs:
        for f in sorted(os.listdir(d)):
            if not f.endswith(".json") or f.endswith(".error.json"):
                if f.endswith(".error.json"):
                    tot["infra_errors"] += 1
                continue
            s = json.load(open(os.path.join(d, f)))
            gen = sum(len(t["gen_tokens"]) for t in s["turns"])
            n_calls = sum(len(t["calls"]) for t in s["turns"])
            bad = sum(1 for t in s["turns"] for c in t["calls"] if not c["result"]["ok"])
            unknown = sum(1 for t in s["turns"] for c in t["calls"] if c["result"]["output"].startswith("error: unknown"))
            tool_ms = sum(c["result"].get("ms", 0) for t in s["turns"] for c in t["calls"])
            gen_ms = sum(t.get("gen_ms", 0) for t in s["turns"])
            problems = sum(len(t["problems"]) for t in s["turns"])
            for t in s["turns"]:
                for c in t["calls"]:
                    calls[c["name"]] += 1
            longest = max((len(t["gen_tokens"]) for t in s["turns"]), default=0)
            rows.append({"session": s["session_id"], "type": s["task"]["type"], "turns": len(s["turns"]), "gen": gen,
                         "longest_turn": longest, "seq": s["sequence_len"], "end": s["end"],
                         "verified": s["verify"]["passed"], "calls": n_calls, "failed_calls": bad,
                         "unknown_tool": unknown, "problems": problems, "tool_s": round(tool_ms / 1000, 1),
                         "gen_s": round(gen_ms / 1000, 1), "wall_s": s["seconds"]})
    for r in rows:
        tot["sessions"] += 1
        tot["verified"] += bool(r["verified"])
        for k in ("turns", "gen", "seq", "calls", "failed_calls", "unknown_tool", "problems"):
            tot[k] += r[k]
        tot["end_" + r["end"]] += 1
    if a.md:
        keys = ["session", "type", "turns", "gen", "longest_turn", "seq", "end", "verified", "calls", "failed_calls",
                "tool_s", "wall_s"]
        print("| " + " | ".join(keys) + " |")
        print("|" + "---|" * len(keys))
        for r in rows:
            print("| " + " | ".join(str(r[k]) for k in keys) + " |")
    else:
        for r in rows:
            print(json.dumps(r))
    by_type = collections.defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    summary = {**tot, "generated_share_of_sequence": round(tot["gen"] / max(tot["seq"], 1), 3),
               "calls_by_tool": dict(calls.most_common()),
               "per_type": {k: {"sessions": len(v), "verified": sum(bool(x["verified"]) for x in v),
                                "median_gen": statistics.median(x["gen"] for x in v),
                                "median_turns": statistics.median(x["turns"] for x in v),
                                "median_wall_s": statistics.median(x["wall_s"] for x in v)}
                            for k, v in by_type.items()}}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
