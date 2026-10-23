"""Evaluate sampled solutions offline with LiveCodeBench's own test runner (evaluate_generations, 6 s per test),
one CSV row per problem, file and attempt (the columns of ../task_results.py; ../paired_tasks.py reads them).

usage:
  python evaluate.py PROBLEMS.jsonl OUT_DIR FILE=RAW.jsonl [FILE=RAW.jsonl ...] [--procs 16]
  python evaluate.py PROBLEMS.jsonl OUT_DIR --replay LABEL=SUBMISSION.json [--procs 16]

The second form re-grades a LiveCodeBench leaderboard submission (its *_eval_all.json, with the code, the raw
answers and LiveCodeBench's own verdicts): our extraction must give the submitted code from the raw answers, and our
verdicts must match theirs. It validates the extraction and evaluation path without a model.

Model-written code runs here: start this inside a sandbox without network (sandbox.sh, or a container with
--network none). The script refuses to run while it can reach the internet, unless --allow-network is given.
"""
import argparse
import json
import os
import socket
from collections import defaultdict

from lcb_common import DIFFICULTIES, code_of, evaluate_generations, load_problems
from task_results import summarize, write_csv

EXTRA = ["platform", "contest_date", "no_code", "lcb_error_code", "lcb_error"]
BATCH_BYTES = 2e9  # test data handed to the worker processes at once (tests x programs)


def network_reachable():
    for host in [("1.1.1.1", 443), ("8.8.8.8", 53)]:
        try:
            socket.create_connection(host, timeout=3).close()
            return True
        except OSError:
            pass
    return False


def grade(batch, procs):
    """LiveCodeBench's verdicts for [(problem, [program, ...]), ...]: per problem a list of (pass, error_code,
    error_message). The programs of all problems run in one pool, as codegen_metrics does; an empty program (no
    code block in the answer) is not run."""
    samples, programs, where = [], [], []
    out = []
    for k, (problem, codes) in enumerate(batch):
        out.append([(False, None, "no code")] * len(codes))
        sample = problem.get_evaluation_sample()
        for i, c in enumerate(codes):
            if c:
                samples.append(sample)
                programs.append([c])
                where.append((k, i))
    if samples:
        results, metadata = evaluate_generations(samples, programs, num_process_evaluate=procs, timeout=6)
        for j, (k, i) in enumerate(where):
            res, meta = results[j][0], metadata[j][0]
            out[k][i] = (all(r > 0 for r in res), meta.get("error_code"), meta.get("error_message"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problems")
    ap.add_argument("out_dir")
    ap.add_argument("raw", nargs="*", help="FILE=RAW.jsonl from run.py")
    ap.add_argument("--replay", action="append", default=[], help="LABEL=SUBMISSION.json")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--allow-network", action="store_true")
    args = ap.parse_args()
    if network_reachable() and not args.allow_network:
        raise SystemExit("network reachable: run this in a sandbox without network (sandbox.sh)")

    # problem -> [(file, record, code, think_open, reference)]
    work = defaultdict(list)
    extraction = [0, 0]
    for spec in args.raw:
        file, path = spec.split("=", 1)
        for r in map(json.loads, open(path)):
            code, think_open = code_of(r.get("content", ""))
            work[r["id"]].append((file, r, code, think_open, None))
    for spec in args.replay:
        label, path = spec.split("=", 1)
        for p in json.load(open(path)):
            for i, (output, code, graded) in enumerate(zip(p["output_list"], p["code_list"], p["graded_list"])):
                ours, _ = code_of(output)
                extraction[0] += 1
                extraction[1] += ours == code
                r = {"id": p["question_id"], "attempt": i + 1, "seed": None, "content": output}
                work[p["question_id"]].append((label, r, code, False, graded))

    rows, agree, disagree = [], defaultdict(int), []
    ids = set(work)

    def flush(batch):
        for (p, _), verdicts in zip(batch, grade(batch, args.procs)):
            for (file, r, code, think_open, ref), (ok, ecode, emsg) in zip(work[p.question_id], verdicts):
                text = r.get("content", "")
                rows.append({
                    "lane": "livecodebench", "task_id": p.question_id, "cluster": p.contest_id,
                    "category": p.difficulty.value, "file": file, "attempt": r["attempt"], "seed": r.get("seed"),
                    "pass": int(ok), "malformed": int(not code),
                    "runaway": int(r.get("stop_type") == "limit" or think_open), "error_type": "" if ok else emsg,
                    "prompt_tokens": r.get("tokens_evaluated"), "output_tokens": r.get("tokens_predicted"),
                    "reasoning_chars": len(text) - len(text.rsplit("</think>", 1)[-1]),
                    "stop_type": r.get("stop_type") or "", "platform": p.platform.value,
                    "contest_date": p.contest_date.date().isoformat(), "no_code": int(not code),
                    "lcb_error_code": "" if ecode is None else ecode, "lcb_error": emsg or ""})
                if ref is not None:
                    agree[(file, bool(ref), ok)] += 1
                    if bool(ref) != ok:
                        disagree.append((file, p.question_id, r["attempt"], bool(ref), ok, emsg))
        print(f"{len({r['task_id'] for r in rows})} problems graded", flush=True)

    batch, size = [], 0
    for p in load_problems(args.problems, ids=ids):
        codes = [it[2] for it in work[p.question_id]]
        batch.append((p, codes))
        size += len(p.get_evaluation_sample()["input_output"]) * sum(map(bool, codes))
        if size > BATCH_BYTES:
            flush(batch)
            batch, size = [], 0
    if batch:
        flush(batch)
    missing = {it[1]["id"] for v in work.values() for it in v if it[4] is None} - {r["task_id"] for r in rows}
    if missing:
        print(f"{len(missing)} problems of the raw outputs are not in {args.problems}: {sorted(missing)[:5]} ...")

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, "per_attempt.csv"), rows, EXTRA)
    text = "All problems\n\n" + summarize(rows, order=DIFFICULTIES)
    recent = [r for r in rows if r["contest_date"] >= "2025-02-01"]
    if recent:
        text += "\nProblems from 2025-02-01\n\n" + summarize(recent, order=DIFFICULTIES)
    if args.replay:
        text += f"\nExtraction: {extraction[1]} of {extraction[0]} answers give the submitted code\n\n"
        text += "| Submission | LiveCodeBench pass, ours pass | LCB pass, ours fail | LCB fail, ours pass | LCB fail, ours fail |"
        text += "\n|---|---|---|---|---|\n"
        for label in sorted({k[0] for k in agree}):
            text += (f"| {label} | {agree[(label, True, True)]} | {agree[(label, True, False)]} | "
                     f"{agree[(label, False, True)]} | {agree[(label, False, False)]} |\n")
        with open(os.path.join(args.out_dir, "replay_disagreements.tsv"), "w") as f:
            f.write("submission\tproblem\tsample\tlcb_pass\tour_pass\tour_error\n")
            for d in disagree:
                f.write("\t".join(map(str, d)) + "\n")
    open(os.path.join(args.out_dir, "summary.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
