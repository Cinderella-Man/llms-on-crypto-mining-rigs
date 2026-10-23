"""Score raw single-turn outputs with BFCL v4's AST checker and (ir)relevance logic, one CSV row per task, file and
attempt, plus a summary per file (pass@1 = mean over tasks of the mean over attempts).

usage: python score.py TASKS.jsonl OUT_DIR FILE=RAW.jsonl [FILE=RAW.jsonl ...]
Writes OUT_DIR/per_attempt.csv (the columns of ../task_results.py, which ../paired_tasks.py reads) and
OUT_DIR/summary.md.
"""
import json
import os
import sys

from bfcl_common import (MODEL, SINGLE_TURN, Language, ast_checker, decode_ast, is_empty_output,
                         is_function_calling_format_output, live_cluster, split_output)
from task_results import summarize, write_csv


def check(t, text):
    """(pass, parsed, error_type), with the verdict BFCL's eval_runner would give."""
    parsed = split_output(text, t["tools"])
    cat = t["category"]
    try:
        decoded = decode_ast(parsed)
    except ValueError:
        decoded = None  # undecodable: no call for (ir)relevance, a decoder failure for AST tasks
    if "irrelevance" in cat:
        ok = decoded is None or is_empty_output(decoded)
        return ok, parsed, None if ok else "irrelevance_error:decoder_success"
    if cat == "live_relevance":
        ok = decoded is not None and not is_empty_output(decoded)
        return ok, parsed, None if ok else "relevance_error:decoder_failed"
    if decoded is None:
        return False, parsed, "ast_decoder:decoder_failed"
    if not is_function_calling_format_output(decoded):
        return False, parsed, "ast_decoder:decoder_wrong_output_format"
    r = ast_checker(t["functions"], decoded, t["ground_truth"], Language.PYTHON, cat, MODEL)
    return bool(r["valid"]), parsed, None if r["valid"] else r.get("error_type")


def row(t, file, r):
    text = r.get("content", "")
    ok, parsed, err = check(t, text)
    if r.get("error"):
        ok, err = False, r["error"]
    runaway = r.get("stop_type") == "limit"
    malformed = bool(parsed["malformed"] or parsed["unknown_function"] or parsed["unknown_arg"]
                     or parsed["call_in_think"])
    return {"lane": "bfcl_single_turn", "task_id": t["id"], "cluster": live_cluster(t["id"]),
            "category": t["category"], "file": file, "attempt": r["attempt"], "seed": r["seed"], "pass": int(ok),
            "malformed": int(malformed), "runaway": int(runaway), "error_type": err or "",
            "prompt_tokens": r.get("tokens_evaluated") or r.get("n_prompt"),
            "output_tokens": r.get("tokens_predicted"), "reasoning_chars": len(parsed["reasoning"]),
            "stop_type": r.get("stop_type") or "", "n_calls": len(parsed["calls"]),
            "undecodable": int(parsed["malformed"]), "unknown_function": parsed["unknown_function"],
            "unknown_arg": parsed["unknown_arg"], "call_in_think": int(parsed["call_in_think"]),
            "extra_think_close": int(parsed["extra_think_close"])}


def write(out_dir, rows, order=None, extra=()):
    os.makedirs(out_dir, exist_ok=True)
    write_csv(os.path.join(out_dir, "per_attempt.csv"), rows, extra)
    text = summarize(rows, order=order)
    open(os.path.join(out_dir, "summary.md"), "w").write(text)
    print(text)


def main():
    tasks = {t["id"]: t for t in map(json.loads, open(sys.argv[1]))}
    rows = []
    for spec in sys.argv[3:]:
        file, path = spec.split("=", 1)
        for r in map(json.loads, open(path)):
            if r["id"] in tasks:
                rows.append(row(tasks[r["id"]], file, r))
    write(sys.argv[2], rows, order=SINGLE_TURN)


if __name__ == "__main__":
    main()
