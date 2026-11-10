#!/usr/bin/env python3
"""exact_table.py: the tables of the draft line's exactness check (registry Q1175) from exact_client.py's files.

  python3 exact_table.py <folder with replies_<arm>.json and requests_<plain>.jsonl> --out <folder> \
      [--plain nodraft] [--draft draft] [--line 0.05]

Writes, with no reply text and no token ids beyond the two candidates at each divergence point:
  divergence.tsv       per depth: the first token where the two replies part, the two candidates, and the lead between
                       them in log-probability as each line saw it while generating and as the plain line gives it for
                       the common tokens read back as a prompt; whether each line wrote the same tokens on its second
                       request and, if not, where it parted from its first and by what lead
  teacher_forced.tsv   per depth and reply: tokens scored on the plain line, how many are not its first choice, the
                       largest lead lost and where; a last row per reply sums the depths
  off_first_choice.tsv every scored token that is not the plain line's first choice, with the lead it loses by
  path_shift.tsv       per depth and pair of paths: how far the log-probability of the same token after the same
                       tokens moves between two ways of computing it (the plain line generating against the plain
                       line teacher-forced on its own reply; the draft line generating against the plain line, on
                       the common tokens and on the draft line's reply teacher-forced)
  summary.md           the same as Markdown, with the verdict per depth
A lead is the difference of two log-probabilities at one position (equal to the difference of the two logits). --line
is the lead above which a difference is not called rounding. End-of-text tokens are left out of a first choice, since
every request ignores them. The server lists no probabilities for a token whose text ends inside a character; such a
position counts by the token the server wrote there.
"""
import argparse
import json
import math
import os
import statistics

ap = argparse.ArgumentParser()
ap.add_argument('folder')
ap.add_argument('--out', required=True)
ap.add_argument('--plain', default='nodraft')
ap.add_argument('--draft', default='draft')
ap.add_argument('--line', type=float, default=0.05)
A = ap.parse_args()


def load(name):
    with open(os.path.join(A.folder, name)) as f:
        return json.load(f)


def lp_of(pos, tid):
    """the log-probability of token tid among a position's listed tokens, or None"""
    return next((lp for t, lp, _ in pos['top'] if t == tid), None)


def piece_of(pos, tid):
    return next((json.dumps(p, ensure_ascii=False) for t, _, p in pos['top'] if t == tid), '')


def f4(x):
    return '' if x is None else f'{x:.4f}'


def lead(pos, a, b):
    """lp(a) - lp(b) at a position; None when one of them is not listed"""
    la, lb = lp_of(pos, a), lp_of(pos, b)
    return None if la is None or lb is None else la - lb


def prepare(rep):
    """rep['pa'][i]: the probabilities the second request listed for its token i, or None (the server lists none for a
    token whose text ends inside a character). rep['v']: positions 0..v stand after the first request's tokens, v
    being where the second request parts from the first (the reply's length when it does not)."""
    ids, again, probs = rep['ids'], rep['genp_ids'], rep['probs']
    pa, j = [], 0
    for t in again:
        if j < len(probs) and probs[j]['id'] == t:
            pa.append(probs[j])
            j += 1
        else:
            pa.append(None)
    rep['pa'] = pa
    rep['v'] = next((i for i, (x, y) in enumerate(zip(ids, again)) if x != y), min(len(ids), len(again)))
    return rep


def at(rep, i):
    return rep['pa'][i] if i <= rep['v'] and i < len(rep['pa']) else None


plain = {int(d): prepare(r) for d, r in load(f'replies_{A.plain}.json')['replies'].items()}
draft = {int(d): prepare(r) for d, r in load(f'replies_{A.draft}.json')['replies'].items()}
# Greedy sampling writes the most probable token it may write, so a token listed above the one a line wrote was not
# allowed there: the end-of-text tokens, which every request here ignores. They are left out of a first choice.
banned = set()
for rep in list(plain.values()) + list(draft.values()):
    for pos in rep['probs']:
        for tid, _, _ in pos['top']:
            if tid == pos['id']:
                break
            banned.add(tid)

# The teacher-forced requests of the plain line: per depth and reply, one row per scored token
scored = {}
with open(os.path.join(A.folder, f'requests_{A.plain}.jsonl')) as f:
    for raw in f:
        r = json.loads(raw)
        if r['kind'] != 'tf' or r['reply'] not in (A.plain, A.draft):
            continue
        want = (plain if r['reply'] == A.plain else draft)[r['depth']]['ids'][r['i']]
        if r['probs']:
            top = [t for t in r['probs'][0]['top'] if t[0] not in banned]
            want_lp = next((lp for tid, lp, _ in top if tid == want), None)
            row = {'first': top[0][0], 'want_lp': want_lp, 'listed': want_lp is not None, 'probs': True,
                   'pos': r['probs'][0],
                   'end_of_text_first': r['probs'][0]['top'][0][0] in banned,
                   'lead_lost': 0.0 if top[0][0] == want else top[0][1] - (want_lp if want_lp is not None else top[-1][1])}
        else:  # no probabilities listed: the token the server wrote is its first choice
            first = r['ids'][0] if r['ids'] else None
            row = {'first': first, 'want_lp': None, 'listed': first == want, 'probs': False, 'end_of_text_first': False,
                   'pos': None,
                   'lead_lost': 0.0 if first == want else None}
        scored.setdefault(r['depth'], {}).setdefault(r['reply'], []).append(
            {'i': r['i'], 'want': want, 'is_first': row['first'] == want, **row})
os.makedirs(A.out, exist_ok=True)

div_cols = ['depth', 'tokens', 'first_differing_token', 'plain_token_id', 'plain_piece', 'draft_token_id', 'draft_piece',
            'plain_generating_lp_plain', 'plain_generating_lp_draft', 'plain_generating_lead_of_plain',
            'draft_generating_lp_draft', 'draft_generating_lp_plain', 'draft_generating_lead_of_draft',
            'read_back_tokens', 'read_back_first_choice', 'read_back_lp_plain', 'read_back_lp_draft',
            'read_back_lead_of_plain', 'plain_reply_same_as_earlier_run', 'draft_reply_same_as_earlier_run',
            'plain_second_request_parts_at', 'plain_second_request_lead', 'draft_second_request_parts_at',
            'draft_second_request_lead']
tf_cols = ['depth', 'reply', 'first_token_scored', 'tokens_scored', 'end_of_text_most_probable', 'no_probabilities',
           'not_first_choice', 'above_line', 'largest_lead_lost', 'at_token', 'lead_unknown',
           'lead_lost_at_first_differing_token']
off_cols = ['depth', 'reply', 'token', 'lead_lost', 'note']
shift_cols = ['depth', 'pair', 'tokens', 'exactly_equal', 'median_abs_shift', 'p90_abs_shift', 'max_abs_shift',
              'at_token', 'mean_kld_over_listed']
div_rows, tf_rows, off_rows, shift_rows, verdicts, totals = [], [], [], [], [], {}


def kld(pa, pb):
    """sum of p_a (lp_a - lp_b) over the tokens both positions list: the KL divergence cut to the listed tokens"""
    lb = {t: lp for t, lp, _ in pb['top']}
    return sum(math.exp(lp) * (lp - lb[t]) for t, lp, _ in pa['top'] if t in lb)


def shift_row(d, pair, triples, lists=()):
    vals = [(abs(a - b), i) for a, b, i in triples if a is not None and b is not None]
    if not vals:
        return
    klds = [kld(a, b) for a, b in lists]
    s = sorted(v for v, _ in vals)
    top = max(vals)
    shift_rows.append({'depth': d, 'pair': pair, 'tokens': len(s), 'exactly_equal': sum(v == 0 for v in s),
                       'median_abs_shift': f4(statistics.median(s)), 'p90_abs_shift': f4(s[int(0.9 * (len(s) - 1))]),
                       'max_abs_shift': f4(top[0]), 'at_token': top[1],
                       'mean_kld_over_listed': f4(statistics.mean(klds)) if klds else ''})


for d in sorted(plain):
    if d not in draft:
        continue
    p, q = plain[d], draft[d]
    n = min(len(p['ids']), len(q['ids']))
    c = next((i for i in range(n) if p['ids'][i] != q['ids'][i]), n)
    row = {'depth': d, 'tokens': n, 'first_differing_token': 'none' if c == n else c,
           'plain_reply_same_as_earlier_run': {True: 'yes', False: 'no'}[p['ids_md5'] == p['ref_md5']] if p.get('ref_md5') else '',
           'draft_reply_same_as_earlier_run': {True: 'yes', False: 'no'}[q['ids_md5'] == q['ref_md5']] if q.get('ref_md5') else '',
           'plain_second_request_parts_at': 'nowhere' if p['genp_same_ids'] else p['v'],
           'draft_second_request_parts_at': 'nowhere' if q['genp_same_ids'] else q['v']}
    for key, rep in (('plain', p), ('draft', q)):
        pos = at(rep, rep['v'])
        if not rep['genp_same_ids'] and pos:  # the second request's lead of its token over the first's
            row[f'{key}_second_request_lead'] = f4(lead(pos, rep['genp_ids'][rep['v']], rep['ids'][rep['v']]))
    leads = []
    if c < n:
        a, b = p['ids'][c], q['ids'][c]
        row.update({'plain_token_id': a, 'draft_token_id': b})
        pos = at(p, c)
        if pos:
            row.update({'plain_piece': piece_of(pos, a), 'draft_piece': piece_of(pos, b),
                        'plain_generating_lp_plain': f4(lp_of(pos, a)), 'plain_generating_lp_draft': f4(lp_of(pos, b)),
                        'plain_generating_lead_of_plain': f4(lead(pos, a, b))})
            leads.append(lead(pos, a, b))
        pos = at(q, c)
        if pos:
            row.update({'draft_generating_lp_draft': f4(lp_of(pos, b)), 'draft_generating_lp_plain': f4(lp_of(pos, a)),
                        'draft_generating_lead_of_draft': f4(lead(pos, b, a))})
            row['draft_piece'] = row.get('draft_piece') or piece_of(pos, b)
            row['plain_piece'] = row.get('plain_piece') or piece_of(pos, a)
            leads.append(lead(pos, b, a))
        if p.get('probe'):
            pos = p['probe']
            first = next(t[0] for t in pos['top'] if t[0] not in banned)
            row.update({'read_back_tokens': p.get('probe_prompt_n', ''),
                        'read_back_first_choice': 'plain' if first == a else 'draft' if first == b else f'another ({first})',
                        'read_back_lp_plain': f4(lp_of(pos, a)), 'read_back_lp_draft': f4(lp_of(pos, b)),
                        'read_back_lead_of_plain': f4(lead(pos, a, b))})
            leads.append(lead(pos, a, b))
    div_rows.append(row)
    if c > 0:
        shift_row(d, f'{A.draft} generating against {A.plain} generating, the common tokens',
                  [(at(p, i)['lp'], at(q, i)['lp'], i) for i in range(c) if at(p, i) and at(q, i)],
                  [(at(q, i), at(p, i)) for i in range(c) if at(p, i) and at(q, i)])

    worst_tf = None
    for name, rows in (scored.get(d) or {}).items():
        off = [x for x in rows if not x['is_first']]
        worst = max((x for x in off if x['lead_lost'] is not None), key=lambda x: x['lead_lost'], default=None)
        at_c = next((x for x in rows if x['i'] == c), None)
        counts = {'tokens_scored': len(rows), 'end_of_text_most_probable': sum(x['end_of_text_first'] for x in rows),
                  'no_probabilities': sum(not x['probs'] for x in rows), 'not_first_choice': len(off),
                  'above_line': sum(x['lead_lost'] is not None and x['lead_lost'] > A.line for x in off),
                  'lead_unknown': sum(x['lead_lost'] is None for x in off)}
        tf_rows.append({'depth': d, 'reply': name, 'first_token_scored': rows[0]['i'], **counts,
                        'largest_lead_lost': f4(worst['lead_lost']) if worst else '',
                        'at_token': worst['i'] if worst else '',
                        'lead_lost_at_first_differing_token': f4(at_c['lead_lost']) if at_c else ''})
        t = totals.setdefault(name, {'depth': 'all', 'reply': name, 'worst': None})
        for k, v in counts.items():
            t[k] = t.get(k, 0) + v
        if worst and (t['worst'] is None or worst['lead_lost'] > t['worst'][0]):
            t['worst'] = (worst['lead_lost'], f"{worst['i']} at depth {d}")
        for x in off:
            off_rows.append({'depth': d, 'reply': name, 'token': x['i'], 'lead_lost': f4(x['lead_lost']),
                             'note': '' if x['listed'] else 'no probabilities listed' if not x['probs']
                             else 'not among the listed tokens: a lower bound'})
        if name != A.plain and worst and (worst_tf is None or worst['lead_lost'] > worst_tf):
            worst_tf = worst['lead_lost']
        gen = p if name == A.plain else q  # the line that wrote this reply, while it generated
        pair = (f'{A.plain} generating against {A.plain} teacher-forced, its own reply' if name == A.plain else
                f'{A.draft} generating against {A.plain} teacher-forced, the reply of {A.draft}')
        shift_row(d, pair, [(x['want_lp'], at(gen, x['i'])['lp'], x['i']) for x in rows if at(gen, x['i'])],
                  [(at(gen, x['i']), x['pos']) for x in rows if at(gen, x['i']) and x['pos']])

    known = [abs(x) for x in leads if x is not None]
    if c == n:
        verdicts.append((d, 'the same tokens in both arms'))
    elif not known:
        verdicts.append((d, 'the other candidate is not among the listed tokens: a lead beyond the list'))
    else:
        big = max(known)
        text = (f'a near-tie (the largest lead between the two candidates {big:.4f})' if big <= A.line
                else f'a lead beyond rounding ({big:.4f} between the two candidates)')
        if worst_tf is not None:
            text += f'; teacher-forced, the largest lead a token of the draft line\'s reply loses is {worst_tf:.4f}'
        verdicts.append((d, text))
for t in totals.values():
    w = t.pop('worst')
    tf_rows.append({**t, 'largest_lead_lost': f4(w[0]) if w else '', 'at_token': w[1] if w else ''})


def write_tsv(name, cols, rows):
    with open(os.path.join(A.out, name), 'w') as f:
        f.write('\t'.join(cols) + '\n')
        for r in rows:
            f.write('\t'.join(str(r.get(c, '')) for c in cols) + '\n')


def md_table(cols, rows):
    out = ['| ' + ' | '.join(c.replace('_', ' ') for c in cols) + ' |', '|' + '---|' * len(cols)]
    out += ['| ' + ' | '.join(str(r.get(c, '')).replace('|', '\\|') for c in cols) + ' |' for r in rows]
    return '\n'.join(out)


write_tsv('divergence.tsv', div_cols, div_rows)
write_tsv('teacher_forced.tsv', tf_cols, tf_rows)
write_tsv('off_first_choice.tsv', off_cols, off_rows)
write_tsv('path_shift.tsv', shift_cols, shift_rows)
with open(os.path.join(A.out, 'summary.md'), 'w') as f:
    f.write(f'# The line with the draft ({A.draft}) against the line without it ({A.plain}): tables\n\n'
            f'The line for rounding: {A.line} in log-probability.\n\n')
    f.write('## Verdict per depth\n\n' + '\n'.join(f'- depth {d:,}: {v}' for d, v in verdicts) + '\n\n')
    f.write('## Where the two replies part\n\n' + md_table(div_cols, div_rows) + '\n\n')
    f.write('## Both replies teacher-forced on the plain line\n\n' + md_table(tf_cols, tf_rows) + '\n\n')
    f.write('## The same token after the same tokens, two paths\n\n' + md_table(shift_cols, shift_rows) + '\n\n')
    f.write('## Every token that is not the plain line\'s first choice\n\n' + md_table(off_cols, off_rows) + '\n')
print('\n'.join(f'depth {d}: {v}' for d, v in verdicts))
