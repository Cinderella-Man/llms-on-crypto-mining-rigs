"""gen_without_task.py DIR TASK ARM...: generation and prompt tok/s per arm over all timed turns and without TASK."""
import csv, os, sys
d, task, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
print("arm\tturns\tgen_tok_s\tturns_wo\tgen_tok_s_wo\tprompt_tok_s_wo\ttask_turns\ttask_gen_tokens\ttask_gen_tok_s\ttask_max_prompt_ctx")
for a in arms:
    rows = []
    for t in csv.DictReader(open(os.path.join(d, "agent", a, "turns.csv"))):
        try:
            rows.append((t["task_id"], int(t["predicted_n"]), float(t["predicted_ms"]), int(t["prompt_n"]), float(t["prompt_ms"]), int(t["prompt_n"]) + int(t["cache_n"] or 0)))
        except (TypeError, ValueError):
            continue
    g = lambda rs, i, j: sum(r[i] for r in rs) / sum(r[j] for r in rs) * 1000
    wo = [r for r in rows if task not in r[0]]
    tk = [r for r in rows if task in r[0]]
    print(f"{a}\t{len(rows)}\t{g(rows,1,2):.2f}\t{len(wo)}\t{g(wo,1,2):.2f}\t{g(wo,3,4):.1f}\t{len(tk)}\t{sum(r[1] for r in tk)}\t{g(tk,1,2) if tk else 0:.2f}\t{max((r[5] for r in tk), default=0)}")
