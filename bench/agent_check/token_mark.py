"""Price run.py's generated-token mark on recorded runs before any new run, and sum up arms run with it.

    python3 token_mark.py price --runs RUNS.tsv --out DIR [--marks 16384,24576,32768,49152,65536] [--tries 2]
    python3 token_mark.py price ARM_DIR … --out DIR
    python3 token_mark.py arms RESULTS [--arms a,b]

price reads every scored attempt of the runs (an arm's folder as run.py writes it: <task>/a<k>/result.json and
events.jsonl). RUNS.tsv names them: columns run, model, family, machine (rig or other) and dir (relative to the
file's folder unless absolute); folders given on the command line are their own run. Per attempt it rebuilds the count
run.py's --gen-mark watches from pi's events: the server's token count of every finished reply plus one per streamed
piece of the reply in flight. Per mark it then finds the event at which that count reaches the mark, and with it:
- the attempts the mark stops: a passed one is a pass lost (its workspace at the stop is counted a fail), a failed
  one a failure cut short;
- the tokens saved: what the attempt generated after the stop (within a finished reply at that reply's own tokens
  per piece; a reply cut off by a cap counts one token per piece, as the mark does);
- on rig runs, the minutes saved: the attempt's wall time after the stop;
- est., with --tries fresh tries per attempt (run.py's --mark-tries, a budget of tries × mark): per task, a share a of
  the group's attempts passes under the mark and a share s reaches it, the tries are taken as independent draws, and
  the expected passes are a (1 - s^N) / (1 - s), the tokens and minutes the mean cost of a try cut at the mark times
  (1 - s^N) / (1 - s); the last try stopped at the mark is counted a fail.
Recorded attempts ran under their own caps (turns, minutes, max_tokens per reply): what they would have generated past
a cap is not known, so a mark above the caps' reach saves nothing here (capped counts the attempts a cap stopped).
Writes DIR/attempts.tsv (one row per attempt, the stop per mark), DIR/price.tsv and DIR/price.md (per model and per
family and mark).

arms prints, per arm of a results folder, the passes, the try that passed (run.py's mark_try), the tries stopped at
the mark, and the tokens and minutes of all tries, so arms of equal budget compare on tasks solved per million
tokens and per hour.
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PIECE = re.compile(rb'"assistantMessageEvent": \{"type": "(\w+)"')
TIME = re.compile(rb'^\{"t": ([0-9.]+)')
MARKS = "16384,24576,32768,49152,65536"


def timeline(path):
    """(points, replies, cut): points = [(t, tokens of finished replies, pieces of the reply in flight)] at every
    event that changes the count; replies = [(pieces, tokens)] of the finished replies; cut = the pieces of a reply
    still streaming when pi stopped."""
    points, replies, done, live = [], [], 0, 0
    with open(path, "rb") as f:
        for line in f:
            m = PIECE.search(line) if b'"message_update"' in line[:60] else None
            if m:
                if m[1].endswith(b"_delta"):
                    live += 1
                    points.append((float(TIME.match(line)[1]), done, live))
                continue
            if b'"message_end"' not in line[:60]:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            msg = r["e"].get("message") or {}
            if msg.get("role") != "assistant":
                continue
            out = (msg.get("usage") or {}).get("output")
            out = out if isinstance(out, (int, float)) else 0
            replies.append((live, out))
            done, live = done + out, 0
            points.append((r["t"], done, 0))
    return points, replies, live


def attempt_row(run, res, events, marks):
    """One attempt's tokens, time and, per mark, the stop: tokens at the stop and seconds into the attempt."""
    points, replies, cut = timeline(events)
    pieces = sum(p for p, _ in replies)
    tokens = sum(t for _, t in replies)
    per_piece = tokens / pieces if pieces else 1.0
    # a reply cut off counts one token per piece, as run.py does: the finished replies' rate is no guide when they
    # are short (a lone tool call's markup tokens stream no piece)
    generated = tokens + cut
    row = dict(run=run["run"], model=run["model"], family=run["family"], machine=run["machine"],
               task_id=res["task_id"], attempt=res["attempt"], passed=res["success"], runaway=res.get("runaway", ""),
               turns=res["turns"], generated=round(generated), wall_s=res["wall_s"], pieces=pieces + cut,
               tokens_per_piece=round(per_piece, 4))
    # the tokens per piece of the reply each point belongs to: a finished reply's own, one for a reply cut off
    reply_rate, i = [], 0
    for t, done, live in points:
        if live == 0:
            reply_rate.append(1.0)
            i += 1
        else:
            p, n = replies[i] if i < len(replies) else (0, 0)
            reply_rate.append(n / p if p else 1.0)
    for m in marks:
        stop = next(((t, done + live * rate) for (t, done, live), rate in zip(points, reply_rate)
                     if done + live >= m), None)
        row[f"stop_tokens_{m}"] = "" if stop is None else round(stop[1])
        row[f"stop_s_{m}"] = "" if stop is None else round(stop[0], 2)
    return row


def load_runs(a):
    runs = []
    if a.runs:
        base = Path(a.runs).resolve().parent
        with open(a.runs) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                d = Path(r["dir"])
                runs.append(dict(r, dir=d if d.is_absolute() else base / d))
    for d in a.dirs:
        runs.append(dict(run=Path(d).name, model="", family="", machine="", dir=Path(d)))
    return runs


def gather(runs, marks):
    """The scored attempts of the runs, each once: a report folder that copied an earlier run's attempts lists them
    again, so an attempt already seen (same arm, task, attempt, start and wall time) is skipped."""
    rows, seen, dup = [], set(), 0
    for run in runs:
        for p in sorted(Path(run["dir"]).glob("*/a*/result.json")):
            res = json.loads(p.read_text())
            if res.get("status") != "scored" or not (p.parent / "events.jsonl").exists():
                continue
            key = (res["arm"], res["task_id"], res["attempt"], res.get("started"), res["wall_s"])
            if key in seen:
                dup += 1
                continue
            seen.add(key)
            rows.append(attempt_row(run, res, p.parent / "events.jsonl", marks))
    return rows, dup


def geometric(s, n):
    """1 + s + … + s^(n-1): the expected number of tries when each one reaches the mark with probability s."""
    return n if s >= 1 else (1 - s ** n) / (1 - s)


def price(rows, marks, tries):
    out = []
    for level in ("model", "family"):
        groups = defaultdict(list)
        for r in rows:
            groups[r[level]].append(r)
        for g, rr in sorted(groups.items()):
            rig = [r for r in rr if r["machine"] == "rig"]
            gen = sum(r["generated"] for r in rr)
            minutes = sum(r["wall_s"] for r in rig) / 60
            by_task = defaultdict(list)
            for r in rr:
                by_task[r["task_id"]].append(r)
            for m in marks:
                stopped = [r for r in rr if r[f"stop_tokens_{m}"] != ""]
                saved = sum(r["generated"] - r[f"stop_tokens_{m}"] for r in stopped)
                saved_s = sum(r["wall_s"] - r[f"stop_s_{m}"] for r in stopped if r["machine"] == "rig")
                est_pass = est_gen = est_min = 0.0
                for tr in by_task.values():
                    n = len(tr)
                    over = [r for r in tr if r[f"stop_tokens_{m}"] != ""]
                    s = len(over) / n
                    a_ = sum(r["passed"] for r in tr if r[f"stop_tokens_{m}"] == "") / n
                    tries_n = geometric(s, tries)
                    est_pass += n * a_ * tries_n
                    est_gen += n * tries_n * sum(r[f"stop_tokens_{m}"] if r[f"stop_tokens_{m}"] != ""
                                                 else r["generated"] for r in tr) / n
                    trig = [r for r in tr if r["machine"] == "rig"]
                    if trig:
                        est_min += len(trig) * tries_n * sum(r[f"stop_s_{m}"] if r[f"stop_s_{m}"] != ""
                                                             else r["wall_s"] for r in trig) / len(trig) / 60
                out.append(dict(level=level, group=g, mark=m, attempts=len(rr), passed=sum(r["passed"] for r in rr),
                                capped=sum(bool(r["runaway"]) for r in rr), generated=gen, over_mark=len(stopped),
                                passes_lost=sum(r["passed"] for r in stopped),
                                failures_cut=sum(1 - r["passed"] for r in stopped), tokens_saved=round(saved),
                                tokens_saved_pct=round(100 * saved / gen, 2) if gen else "",
                                rig_attempts=len(rig), rig_minutes=round(minutes, 1),
                                rig_minutes_saved=round(saved_s / 60, 1),
                                rig_minutes_saved_pct=round(100 * saved_s / 60 / minutes, 2) if minutes else "",
                                tries=tries, budget=tries * m, est_passed=round(est_pass, 2),
                                est_generated=round(est_gen), est_rig_minutes=round(est_min, 1) if rig else ""))
    return out


def write_tsv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def md_table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def cmd_price(a):
    marks = [int(x) for x in a.marks.split(",")]
    rows, dup = gather(load_runs(a), marks)
    if not rows:
        sys.exit("no scored attempts with events.jsonl found")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_tsv(out / "attempts.tsv", rows)
    pr = price(rows, marks, a.tries)
    write_tsv(out / "price.tsv", pr)
    pieces = sum(r["pieces"] for r in rows)
    ratio = sum(r["generated"] for r in rows) / pieces if pieces else 0
    cols = ["group", "mark", "attempts", "passed", "capped", "generated", "over_mark", "passes_lost", "failures_cut",
            "tokens_saved", "tokens_saved_pct", "rig_minutes", "rig_minutes_saved", "rig_minutes_saved_pct",
            "budget", "est_passed", "est_generated", "est_rig_minutes"]
    text = [f"{len(rows)} scored attempts of {len({r['run'] for r in rows})} runs ({dup} copies of an attempt "
            f"skipped); marks {a.marks}; {ratio:.3f} tokens per streamed piece over all replies.", "",
            "Per model:", "", md_table([r for r in pr if r["level"] == "model"], cols), "",
            "Per family:", "", md_table([r for r in pr if r["level"] == "family"], cols), ""]
    (out / "price.md").write_text("\n".join(text))
    print("\n".join(text))


def cmd_arms(a):
    root = Path(a.results)
    arms = a.arms.split(",") if a.arms else sorted(p.name for p in root.iterdir() if p.is_dir())
    cols = ["arm", "attempts", "passed", "passed_by_try", "tries_stopped_at_mark", "generated_all",
            "wall_all_min", "passed_per_mtok", "passed_per_hour"]
    rows = []
    for arm in arms:
        rr = [json.loads(p.read_text()) for p in sorted((root / arm).glob("*/a*/result.json"))]
        rr = [r for r in rr if r.get("status") == "scored"]
        if not rr:
            continue
        by_try = defaultdict(int)
        for r in rr:
            if r["success"]:
                by_try[r.get("mark_try", 1)] += 1
        gen = sum(r.get("generated_all", r.get("generated", r.get("usage_output", 0))) for r in rr)
        wall = sum(r.get("wall_all_s", r["wall_s"]) for r in rr) / 60
        ok = sum(r["success"] for r in rr)
        rows.append(dict(arm=arm, attempts=len(rr), passed=ok,
                         passed_by_try=" ".join(f"{k}:{v}" for k, v in sorted(by_try.items())) or "-",
                         tries_stopped_at_mark=sum(r.get("mark_try", 1) - 1 for r in rr), generated_all=gen,
                         wall_all_min=round(wall, 1), passed_per_mtok=round(ok / gen * 1e6, 2) if gen else "",
                         passed_per_hour=round(ok / wall * 60, 2) if wall else ""))
    print(md_table(rows, cols))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("price", help="what a mark would have stopped, lost and saved on recorded runs")
    p.add_argument("dirs", nargs="*", help="arm folders (<task>/a<k>/result.json), each its own run")
    p.add_argument("--runs", help="TSV: run, model, family, machine, dir")
    p.add_argument("--marks", default=MARKS)
    p.add_argument("--tries", type=int, default=2, help="tries per attempt in the estimate (run.py's --mark-tries)")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_price)
    p = sub.add_parser("arms", help="per arm: passes, the try that passed, tokens and minutes of all tries")
    p.add_argument("results")
    p.add_argument("--arms", default="")
    p.set_defaults(func=cmd_arms)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
