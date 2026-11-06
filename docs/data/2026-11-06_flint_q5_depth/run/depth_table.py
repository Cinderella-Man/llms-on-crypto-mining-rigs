#!/usr/bin/env python3
# depth_table.py: the tables of run/depth_sweep.sh's loads, build A against build B, standard library only:
#   python3 -I depth_table.py <sweep OUT folder> <A loads, comma-separated> <B loads> <label A> <label B> <out folder>
# Reads <load>.jsonl (the client's requests) and rig/server-<load>.log (each request's pipelined guesses, from the
# server's "pipelined = G guesses, R right, W wrong" line after its "total time" line), writes <out>/requests.tsv (every
# request) and <out>/table.md: generation tok/s by depth for each load (pass 1 and pass 2, the same request twice),
# the guesses, prompt reading, and the checks (the same prompt ids on every load, N tokens out every time, the
# generated ids per load and per build). Verdict per depth: a load's speed is its two passes' tokens over their time;
# spread = the larger of |A1 - A2| and |B1 - B2|; delta = mean(B) - mean(A); gain if delta > spread, loss if
# delta < -spread, else negligible.
import json, os, re, sys

src, a_loads, b_loads, la, lb, out = sys.argv[1:7]
a_loads, b_loads = a_loads.split(','), b_loads.split(',')
loads = a_loads + b_loads
build = {ld: la if ld in a_loads else lb for ld in loads}
os.makedirs(out, exist_ok=True)
meta, reqs, order = {}, {}, {ld: [] for ld in loads}
for ld in loads:
    for line in open(os.path.join(src, ld + '.jsonl')):
        r = json.loads(line)
        if r['kind'] == 'meta':
            meta[ld] = r
        else:
            reqs[(ld, r['kind'], r['depth'], r['pass'])] = r
            order[ld].append(r)
    # the server log: one "total time" line per completion request, in the client's order
    blocks, cur = [], None
    for line in open(os.path.join(src, 'rig', f'server-{ld}.log'), errors='replace'):
        if 'total time =' in line:
            cur = {'guesses': 0, 'right': 0, 'wrong': 0}
            blocks.append(cur)
        m = re.search(r'pipelined = +(\d+) guesses, (\d+) right, (\d+) wrong', line)
        if m and cur is not None:
            cur.update(guesses=int(m[1]), right=int(m[2]), wrong=int(m[3]))
    if len(blocks) != len(order[ld]):
        sys.exit(f'{ld}: {len(blocks)} requests in the server log, {len(order[ld])} in the client')
    for r, b in zip(order[ld], blocks):
        r['pipe'] = b
depths = meta[loads[0]]['depths']
n_predict = meta[loads[0]]['n_predict']


def g(ld, d, p, key):
    return reqs[(ld, 'decode', d, p)]['timings'][key]


def pipe(ld, d, p):
    return reqs[(ld, 'decode', d, p)]['pipe']


def load_speed(ld, d):
    return sum(g(ld, d, p, 'predicted_n') for p in (1, 2)) * 1000 / sum(g(ld, d, p, 'predicted_ms') for p in (1, 2))


def verdict(a, b):
    spread = max(abs(a[0] - a[1]), abs(b[0] - b[1]))
    delta = sum(b) / len(b) - sum(a) / len(a)
    return spread, delta, 'gain' if delta > spread else 'loss' if delta < -spread else 'negligible'


cols = ['load', 'build', 'kind', 'depth', 'pass', 'prompt_tokens', 'prompt_n', 'prompt_ms', 'prompt_per_second',
        'predicted_n', 'predicted_ms', 'predicted_per_second', 'pipe_guesses', 'pipe_right', 'pipe_wrong', 'wall_s',
        'n_out', 'gen_md5', 'time']
with open(os.path.join(out, 'requests.tsv'), 'w') as f:
    f.write('\t'.join(cols) + '\n')
    for ld in loads:
        for r in order[ld]:
            t = r['timings']
            row = [ld, build[ld], r['kind'], r['depth'], r['pass'], r['prompt_tokens'], t.get('prompt_n'),
                   t.get('prompt_ms'), t.get('prompt_per_second'), t.get('predicted_n'), t.get('predicted_ms'),
                   t.get('predicted_per_second'), r['pipe']['guesses'], r['pipe']['right'], r['pipe']['wrong'],
                   r['wall_s'], r['n_out'], r['gen_md5'], r['time']]
            f.write('\t'.join('' if v is None else str(v) for v in row) + '\n')

started = sorted(loads, key=lambda x: meta[x]['time'])
md = [f'## Generation speed by context depth: {la} against {lb}', '',
      f'Tokens per second as llama-server reports them (`timings.predicted_per_second`), {n_predict} greedy tokens per '
      f'request, each request twice per load (pass 1, pass 2), loads in the order {", ".join(started)}. A load\'s '
      f'speed is its two passes\' tokens over their time. Spread: the larger of the two builds\' differences between '
      'their two loads. Class: gain or loss when the difference of the means is beyond the spread, else negligible.', '',
      '| Depth (tokens) | ' + ' | '.join(f'{build[ld]} {ld}: pass 1, pass 2 → load' for ld in loads)
      + f' | Spread | {lb} − {la} (means) | | Class |',
      '|---:|' + '---|' * len(loads) + '---:|---:|---:|---|']
for d in depths:
    cells = [f'{g(ld, d, 1, "predicted_per_second"):.2f}, {g(ld, d, 2, "predicted_per_second"):.2f} → '
             f'**{load_speed(ld, d):.2f}**' for ld in loads]
    a, b = [load_speed(ld, d) for ld in a_loads], [load_speed(ld, d) for ld in b_loads]
    spread, delta, cls = verdict(a, b)
    md.append(f'| {d:,} | ' + ' | '.join(cells)
              + f' | {spread:.2f} | {delta:+.2f} tok/s | {100 * delta / (sum(a) / len(a)):+.1f} % | {cls} |')
tot = {ld: sum(g(ld, d, p, 'predicted_n') for d in depths for p in (1, 2)) * 1000
       / sum(g(ld, d, p, 'predicted_ms') for d in depths for p in (1, 2)) for ld in loads}
a, b = [tot[ld] for ld in a_loads], [tot[ld] for ld in b_loads]
spread, delta, cls = verdict(a, b)
md.append('| all five | ' + ' | '.join(f'**{tot[ld]:.2f}**' for ld in loads)
          + f' | {spread:.2f} | {delta:+.2f} tok/s | {100 * delta / (sum(a) / len(a)):+.1f} % | {cls} |')

md += ['', '### The pipelined guesses (`--pipe-depth 4`), pass 1 and pass 2: right of all guesses', '',
       '| Depth (tokens) | ' + ' | '.join(f'{build[ld]} {ld}' for ld in loads) + ' |',
       '|---:|' + '---|' * len(loads)]
for d in depths:
    md.append(f'| {d:,} | ' + ' | '.join(
        ', '.join(f'{pipe(ld, d, p)["right"]} of {pipe(ld, d, p)["guesses"]}' for p in (1, 2)) for ld in loads) + ' |')

def read_speed(ld, d):
    return sum(g(ld, d, p, 'prompt_n') for p in (1, 2)) * 1000 / sum(g(ld, d, p, 'prompt_ms') for p in (1, 2))


md += ['', '### Prompt reading at each depth (the whole prompt fresh), pass 1, pass 2 → load, tok/s', '',
       '| Depth (tokens) | Prompt tokens | ' + ' | '.join(f'{build[ld]} {ld}' for ld in loads)
       + f' | Spread | {lb} − {la} (means) | | Class |',
       '|---:|---:|' + '---|' * len(loads) + '---:|---:|---:|---|']
for d in depths:
    a, b = [read_speed(ld, d) for ld in a_loads], [read_speed(ld, d) for ld in b_loads]
    spread, delta, cls = verdict(a, b)
    md.append(f'| {d:,} | {g(loads[0], d, 1, "prompt_n"):,} | ' + ' | '.join(
        ', '.join(f'{g(ld, d, p, "prompt_per_second"):.1f}' for p in (1, 2)) + f' → **{read_speed(ld, d):.1f}**'
        for ld in loads) + f' | {spread:.1f} | {delta:+.1f} tok/s | {100 * delta / (sum(a) / len(a)):+.1f} % | {cls} |')

md += ['', '### Checks', '']
m0 = meta[loads[0]]
same_ids = all(meta[ld]['ids_md5'] == m0['ids_md5'] for ld in loads)
md.append(f'- Prompts: the chat template\'s {m0["head_ids"]} head and {m0["tail_ids"]} tail ids around the code of the '
          f'first {m0["files"]} files; prompt ids per depth '
          + ', '.join(f'{m0["prompt_ids"][str(d)]:,}' for d in depths) + '; '
          + ('the same ids (md5 per depth) on every load.' if same_ids else 'NOT the same ids on every load.'))
bad = [k for k, r in reqs.items() if k[1] == 'decode' and r['timings'].get('predicted_n') != n_predict]
md.append(f'- Tokens out: {n_predict} on every timed request.' if not bad else f'- Tokens out differ from {n_predict}: {bad}')
for name, group in ((la, a_loads), (lb, b_loads)):
    same = all(len({reqs[(ld, 'decode', d, p)]['gen_md5'] for ld in group for p in (1, 2)}) == 1 for d in depths)
    md.append(f'- Generated ids, {name}: ' + ('the same on both passes of both loads at every depth.' if same
                                             else 'not the same on every pass and load.'))
cross = [d for d in depths if reqs[(a_loads[0], 'decode', d, 1)]['gen_md5'] != reqs[(b_loads[0], 'decode', d, 1)]['gen_md5']]
md.append(f'- Generated ids, {la} against {lb}: '
          + (f'differ at depth {", ".join(f"{d:,}" for d in cross)}.' if cross else 'the same at every depth.'))
open(os.path.join(out, 'table.md'), 'w').write('\n'.join(md) + '\n')
print('\n'.join(md))
