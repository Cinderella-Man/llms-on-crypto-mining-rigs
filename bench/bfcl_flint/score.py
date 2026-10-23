"""Score raw outputs with the BFCL v3 AST / relevance logic.

usage: python score.py TASKS.jsonl RAW_DIR OUT_DIR REF_TAG TAG [TAG ...]
Writes OUT_DIR/scores.json, OUT_DIR/scores.md and OUT_DIR/per_task.csv.
"""
import csv
import json
import math
import os
import sys

from bfcl_common import (CATEGORIES, ast_checker, decode_flint, is_empty_output,
                         is_function_calling_format_output)

MODEL = "flint-FC"


def check(t, text):
    """Returns (correct, decoded, error_type) exactly as BFCL's eval_runner would."""
    try:
        decoded = decode_flint(text, t["tools"])
    except Exception as e:  # decoder failure counts as "no function call"
        decoded, err = None, f"decode_error:{e}"
        if "irrelevance" in t["category"]:
            return True, None, None
        return False, None, err
    if "irrelevance" in t["category"]:
        has_call = not is_empty_output(decoded)
        return (not has_call), decoded, ("irrelevance_error:decoder_success" if has_call else None)
    if not is_function_calling_format_output(decoded):
        return False, decoded, "ast_decoder:decoder_wrong_output_format"
    r = ast_checker(t["functions"], decoded, t["ground_truth"], "Python", t["category"], MODEL)
    return bool(r["valid"]), decoded, (None if r["valid"] else r.get("error_type"))


def canon(decoded):
    if not decoded or is_empty_output(decoded):
        return ()
    return tuple(sorted(json.dumps(c, sort_keys=True, ensure_ascii=False) for c in decoded))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    tasks_path, raw_dir, out_dir, ref_tag = sys.argv[1:5]
    tags = sys.argv[5:]
    tasks = [json.loads(l) for l in open(tasks_path)]
    results = {}
    per_task = {}
    for tag in tags:
        raw = {}
        for l in open(os.path.join(raw_dir, f"{tag}.jsonl")):
            r = json.loads(l)
            raw[r["id"]] = r
        rows = {}
        for t in tasks:
            r = raw.get(t["id"])
            if r is None:
                continue
            ok, dec, err = check(t, r.get("content", ""))
            rows[t["id"]] = {
                "category": t["category"], "correct": ok, "error_type": err,
                "calls": canon(dec), "has_call": bool(canon(dec)),
                "stop_type": r.get("stop_type"), "tokens_predicted": r.get("tokens_predicted"),
                "think": "</think>" in r.get("content", ""),
            }
        per_task[tag] = rows

    for tag in tags:
        rows = per_task[tag]
        n = len(rows)
        k = sum(r["correct"] for r in rows.values())
        lo, hi = wilson(k, n)
        res = {
            "n": n, "correct": k, "accuracy": k / n if n else float("nan"),
            "ci95_wilson": [lo, hi],
            "ci95_halfwidth_normal": 1.96 * math.sqrt((k / n) * (1 - k / n) / n) if n else None,
            "by_category": {},
            "stopped_by_limit": sum(1 for r in rows.values() if r["stop_type"] == "limit"),
            "with_think_block": sum(1 for r in rows.values() if r["think"]),
        }
        for cat in CATEGORIES:
            cr = [r for r in rows.values() if r["category"] == cat]
            res["by_category"][cat] = {"n": len(cr), "correct": sum(r["correct"] for r in cr),
                                       "accuracy": (sum(r["correct"] for r in cr) / len(cr)) if cr else None}
        ref = per_task.get(ref_tag)
        if ref is not None and tag != ref_tag:
            common = [i for i in rows if i in ref]
            same_call = sum(rows[i]["calls"] == ref[i]["calls"] for i in common)
            same_dec = sum(rows[i]["has_call"] == ref[i]["has_call"] for i in common)
            b = sum(ref[i]["correct"] and not rows[i]["correct"] for i in common)
            c = sum(rows[i]["correct"] and not ref[i]["correct"] for i in common)
            res["vs_ref"] = {
                "ref": ref_tag, "n_common": len(common),
                "same_tool_call_pct": 100 * same_call / len(common),
                "same_decision_pct": 100 * same_dec / len(common),
                "ref_only_correct": b, "this_only_correct": c,
                "mcnemar_exact_p": mcnemar_exact(b, c),
            }
        results[tag] = res

    os.makedirs(out_dir, exist_ok=True)
    json.dump(results, open(os.path.join(out_dir, "scores.json"), "w"), indent=2)

    with open(os.path.join(out_dir, "per_task.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "category"] + [f"{tag}_{c}" for tag in tags for c in ("correct", "error_type", "calls")])
        for t in tasks:
            row = [t["id"], t["category"]]
            for tag in tags:
                r = per_task[tag].get(t["id"])
                row += ([int(r["correct"]), r["error_type"] or "", " | ".join(r["calls"])] if r else ["", "", ""])
            w.writerow(row)

    short = {"simple": "simple", "multiple": "multiple", "parallel": "parallel",
             "parallel_multiple": "par_mult", "live_simple": "live_simple",
             "live_multiple": "live_mult", "irrelevance": "irrel", "live_irrelevance": "live_irrel"}
    lines = ["| File | n | BFCL correct | 95 % CI | " + " | ".join(short[c] for c in CATEGORIES)
             + " | same call as BF16 | McNemar p vs BF16 |",
             "|---|---|---|---|" + "---|" * len(CATEGORIES) + "---|---|"]
    for tag in tags:
        r = results[tag]
        cats = " | ".join(f"{100 * r['by_category'][c]['accuracy']:.1f}" if r["by_category"][c]["accuracy"] is not None else "—"
                          for c in CATEGORIES)
        vs = r.get("vs_ref")
        lines.append(f"| {tag} | {r['n']} | {100 * r['accuracy']:.1f} % | {100 * r['ci95_wilson'][0]:.1f}–{100 * r['ci95_wilson'][1]:.1f} | {cats} | "
                     + (f"{vs['same_tool_call_pct']:.1f} % | {vs['mcnemar_exact_p']:.2f} |" if vs else "— | — |"))
    open(os.path.join(out_dir, "scores.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    for tag in tags:
        r = results[tag]
        print(tag, "stopped_by_limit", r["stopped_by_limit"], "think_blocks", r["with_think_block"], r.get("vs_ref", ""))


if __name__ == "__main__":
    main()
