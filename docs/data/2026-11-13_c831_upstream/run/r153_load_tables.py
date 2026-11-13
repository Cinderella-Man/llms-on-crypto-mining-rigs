#!/usr/bin/env python3
"""R153 part (3): the tables of r153_load.sh's output folder (standard library only).

Usage: r153_load_tables.py OUT DEST
  OUT   r153_load.sh's OUT: <model>/load-<label>/{mem.csv,mem-after.csv,meminfo.txt,meminfo.log,reply-*.json,console.txt},
        <model>/greedy/{greedy-<model>-code.tsv,<model>-code-<build>.out}
  DEST  a folder for the tables:
        load-memory.tsv   per model and load: up or not, seconds to load, MiB per card 20 s after the server answered and
                          after the two requests, the sum, MemAvailable then and its minimum over the load (meminfo.log)
        load-replies.tsv  per model, load and prompt: tokens generated, md5 of the text, drafted and accepted tokens,
                          the generation speed the server reports (a diagnostic), the first characters
        reply-agree.tsv   per model and prompt: each load's reply against the reference build's (label 'ref'): the same
                          bytes or not, the common prefix in characters and in whitespace-separated words, both lengths
        greedy.tsv        rig_greedy.sh's rows with the common prefix of the two builds' texts
"""
import csv
import glob
import hashlib
import json
import os
import re
import sys


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def words_prefix(a, b):
    wa, wb = a.split(), b.split()
    return common_prefix(wa, wb), len(wa), len(wb)


def read_mem(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 2 and parts[1].isdigit():
                rows.append(int(parts[1]))
    return rows


def main():
    out, dest = sys.argv[1], sys.argv[2]
    os.makedirs(dest, exist_ok=True)
    mem_rows, rep_rows, agree_rows, greedy_rows = [], [], [], []
    replies = {}
    for model in sorted(os.listdir(out)):
        mdir = os.path.join(out, model)
        if not os.path.isdir(mdir):
            continue
        for ldir in sorted(glob.glob(os.path.join(mdir, 'load-*'))):
            label = os.path.basename(ldir)[5:]
            con = open(os.path.join(ldir, 'console.txt'), errors='replace').read() \
                if os.path.exists(os.path.join(ldir, 'console.txt')) else ''
            m = re.search(r'up after (\d+) s', con)
            up = 'up' if m else ('exited' if 'exited while loading' in con else 'no')
            mem = read_mem(os.path.join(ldir, 'mem.csv'))
            after = read_mem(os.path.join(ldir, 'mem-after.csv'))
            avail = ''
            mi = os.path.join(ldir, 'meminfo.txt')
            if os.path.exists(mi):
                for line in open(mi):
                    if line.startswith('MemAvailable:'):
                        avail = line.split()[1]
            mlog = os.path.join(ldir, 'meminfo.log')
            minav = ''
            if os.path.exists(mlog):
                vals = [int(line.split()[1]) for line in open(mlog) if len(line.split()) >= 2 and line.split()[1].isdigit()]
                minav = str(min(vals)) if vals else ''
            mem_rows.append([model, label, up, m.group(1) if m else '', ' '.join(map(str, mem)), sum(mem) if mem else '',
                             ' '.join(map(str, after)), sum(after) if after else '', avail, minav])
            for p in ('code', 'prose'):
                rp = os.path.join(ldir, f'reply-{p}.json')
                if not os.path.exists(rp) or os.path.getsize(rp) == 0:
                    rep_rows.append([model, label, p, 'no reply', '', '', '', '', ''])
                    continue
                try:
                    j = json.load(open(rp))
                except ValueError:
                    rep_rows.append([model, label, p, 'not JSON', '', '', '', '', open(rp, errors='replace').read(80)])
                    continue
                text = j.get('content', '')
                t = j.get('timings', {}) or {}
                replies[(model, label, p)] = text
                rep_rows.append([model, label, p, j.get('tokens_predicted', ''),
                                 hashlib.md5(text.encode()).hexdigest(), t.get('draft_n', ''), t.get('draft_n_accepted', ''),
                                 round(t.get('predicted_per_second', 0), 2) if t else '',
                                 text[:60].replace('\n', '\\n').replace('\t', ' ')])
        g = glob.glob(os.path.join(mdir, 'greedy', 'greedy-*.tsv'))
        if g:
            rows = list(csv.reader(open(g[0]), delimiter='\t'))[1:]
            texts = []
            for r in rows:
                b = r[0].split('-')[2]
                f = glob.glob(os.path.join(mdir, 'greedy', f'*-{b}.out'))
                texts.append(open(f[0], errors='replace').read() if f else '')
            for i, r in enumerate(rows):
                cp = common_prefix(texts[0], texts[i]) if texts else ''
                wp, wa, wb = words_prefix(texts[0], texts[i]) if texts else ('', '', '')
                greedy_rows.append([model] + r + [cp, f'{wp} of {wa}/{wb}'])
    for (model, label, p), text in sorted(replies.items()):
        ref = replies.get((model, 'ref', p))
        if ref is None or label == 'ref':
            continue
        wp, wa, wb = words_prefix(ref, text)
        agree_rows.append([model, p, label, 'same' if ref == text else 'differs', common_prefix(ref, text), len(ref),
                           len(text), wp, wa, wb])

    def write(name, header, rows):
        with open(os.path.join(dest, name), 'w', newline='') as f:
            w = csv.writer(f, delimiter='\t', lineterminator='\n')
            w.writerow(header)
            w.writerows(rows)

    write('load-memory.tsv', ['model', 'load', 'state', 'load_s', 'mib_per_card_20s', 'mib_sum_20s',
                              'mib_per_card_after', 'mib_sum_after', 'memavailable_kb_20s', 'memavailable_kb_min'], mem_rows)
    write('load-replies.tsv', ['model', 'load', 'prompt', 'tokens', 'md5', 'drafted', 'accepted', 'gen_tok_s_diag',
                               'start'], rep_rows)
    write('reply-agree.tsv', ['model', 'prompt', 'load', 'vs_ref', 'common_prefix_chars', 'ref_chars', 'chars',
                              'common_prefix_words', 'ref_words', 'words'], agree_rows)
    write('greedy.tsv', ['model', 'build', 'md5', 'bytes', 'eval', 'common_prefix_chars_vs_first', 'common_words'],
          greedy_rows)


if __name__ == '__main__':
    main()
