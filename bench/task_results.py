"""The per-attempt result table shared by the task harnesses (bfcl_flint single- and multi-turn, livecodebench_flint)
and read by paired_tasks.py: one row per task, file and attempt.

- lane: bfcl_single_turn, bfcl_multi_turn or livecodebench; task_id; cluster (tasks that share a source or
  scenario, for clustered standard errors); category; file (the quantized file's name); attempt; seed;
- pass (0/1); malformed (0/1: an answer the format cannot carry, see each harness); runaway (0/1: stopped by the
  token limit or never closed its reasoning); error_type (the checker's reason for a fail);
- prompt_tokens, output_tokens, reasoning_chars, stop_type, and the tool-call process counts (n_calls, undecodable,
  unknown_function, unknown_arg, call_in_think, extra_think_close; empty for LiveCodeBench).
Each harness may append its own columns after these.
"""
import csv
import math
from collections import defaultdict

COLUMNS = ["lane", "task_id", "cluster", "category", "file", "attempt", "seed", "pass", "malformed", "runaway",
           "error_type", "prompt_tokens", "output_tokens", "reasoning_chars", "stop_type", "n_calls",
           "undecodable", "unknown_function", "unknown_arg", "call_in_think", "extra_think_close"]


def write_csv(path, rows, extra=()):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS + list(extra), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def summarize(rows, by="category", order=None):
    """pass@1 (mean over tasks of the mean over attempts) with a task-level 95 % half-width, the process rates and
    the mean output length, per file and per group."""
    per_task = defaultdict(list)
    for r in rows:
        per_task[(r["file"], r["task_id"], r[by])].append(r)
    files = sorted({k[0] for k in per_task})
    groups = order or sorted({k[2] for k in per_task})
    lines = ["| File | tasks | attempts | pass@1 | ± 95 % | malformed | runaway | mean output tokens | "
             + " | ".join(groups) + " |", "|---" * (8 + len(groups)) + "|"]
    for f in files:
        tasks = {k: v for k, v in per_task.items() if k[0] == f}
        means = [sum(x["pass"] for x in v) / len(v) for v in tasks.values()]
        n = len(means)
        m = sum(means) / n
        se = math.sqrt(sum((x - m) ** 2 for x in means) / (n - 1) / n) if n > 1 else float("nan")
        allr = [x for v in tasks.values() for x in v]
        cells = []
        for g in groups:
            gm = [sum(x["pass"] for x in v) / len(v) for k, v in tasks.items() if k[2] == g]
            cells.append(f"{100 * sum(gm) / len(gm):.1f}" if gm else "—")
        tok = [int(x["output_tokens"]) for x in allr if x["output_tokens"] not in (None, "")]
        lines.append(f"| {f} | {n} | {len(allr)} | {100 * m:.1f} % | {196 * se:.1f} | "
                     f"{100 * sum(x['malformed'] for x in allr) / len(allr):.2f} % | "
                     f"{100 * sum(x['runaway'] for x in allr) / len(allr):.2f} % | "
                     f"{sum(tok) / max(1, len(tok)):.0f} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
