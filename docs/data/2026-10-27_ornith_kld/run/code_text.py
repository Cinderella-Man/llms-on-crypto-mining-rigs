"""Plain text of the agent check's Exercism code, for a code-text KL-divergence base (llama-perplexity tokenizes it with
the model's own tokenizer): each polyglot task's reference solution and its hidden test file, Python and JavaScript, in
task order, each file after a "### <task>/<folder>/<file>" header. MIT (Exercism; the licences are next to the tasks).
Scoring data of the check, used here only to measure a quant's divergence from its reference.

usage: python3 -I code_text.py TASKS_POLYGLOT_DIR OUT.txt
"""
import os
import sys


def main():
    root, out = sys.argv[1:]
    n = chars = 0
    with open(out, "w") as f:
        for task in sorted(os.listdir(root)):
            d = os.path.join(root, task)
            if not os.path.isdir(d):
                continue
            for sub in ("reference", "hidden"):
                p = os.path.join(d, sub)
                for name in sorted(os.listdir(p)) if os.path.isdir(p) else []:
                    text = open(os.path.join(p, name), encoding="utf-8").read()
                    f.write(f"### {task}/{sub}/{name}\n{text}\n\n")
                    n += 1
                    chars += len(text)
    print(f"{n} files, {chars / 1e6:.2f} M characters -> {out}")


if __name__ == "__main__":
    main()
