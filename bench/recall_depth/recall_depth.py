#!/usr/bin/env python3
"""recall_depth.py: does a model served by llama-server still recall what it was told deep in a long prompt?

This project's own version of the seeded recall test in upstream llama.cpp issue #32375 (registry row Q1546). There, on
a Tesla P100 (compute 6.0), a hybrid model answered a 60,240-token prompt with a short echo and none of six planted
facts, while the same binary on the CPU listed all six; the same prompt family passed at 56,460 tokens.

For every target length the script builds one conversation: small talk between a user and an assistant, in which the
user states six facts about themselves, each planted a fixed number of tokens from the end of the prompt (FACTS:
48,000, 32,000, 16,000, 4,000, 1,000 and 300), and a last question that asks for every fact as a list. The
conversation is built backward from the question, so the six depths do not depend on the length: a longer prompt only
has more small talk at its start. A fact deeper than the prompt is long goes to its start, in order.

Lengths and depths are counted by the server itself: its chat template (/apply-template) and tokenizer (/tokenize).
The table gives each prompt's length as the server counted it for the request and each fact's measured depth, from the
fact's first token to the end of the prompt. The small talk is drawn from a fixed pool with a seeded random generator
(SEED), so the same seed and the same model give the same prompts; each prompt's SHA-256 is recorded.

The request is greedy (temperature 0, top-k 1) and reads the whole prompt (cache_prompt false, so a repeat computes
everything again). A fact counts as recalled when its marker word is in the reply's answer (its content, without the
reasoning); a prompt passes with all six. A prompt that misses facts is sent once more (REPEAT_MISS=1) to see whether
the miss repeats, and the run stops after that second try, whatever it gives.

    URL=http://127.0.0.1:8080 OUT=<results folder> python3 recall_depth.py run 8000 56460 60240
    OUT=<results folder> python3 recall_depth.py table > table.md
    URL=... python3 recall_depth.py show 8000            # print one prompt's messages, send nothing

Env:
  URL         the llama-server (default http://127.0.0.1:8080)
  OUT         results folder: prompts/<name>.json.gz (the messages), <name>-<try>.json (request settings, measured
              depths and the full reply), results.tsv (one row a request)
  SEED        the small talk's seed (default 1546)
  CTX         the server's context (default 65536): a reply gets at most CTX minus the prompt, less 16 tokens
  MAX_TOKENS  the reply's cap (default 8192)
  REPEAT_MISS 1 (default): a prompt that misses a fact is sent once more, then the run stops; 0: no repeat, go on
  TIMEOUT_S   one request's timeout (default 7200)
  NAME        the prompts' name prefix (default "recall"): prompt <NAME>-<target length>
"""
import gzip
import hashlib
import json
import os
import random
import sys
import time
import urllib.request

URL = os.environ.get('URL', 'http://127.0.0.1:8080').rstrip('/')
OUT = os.environ.get('OUT', '')
SEED = int(os.environ.get('SEED', '1546'))
CTX = int(os.environ.get('CTX', '65536'))
MAX_TOKENS = int(os.environ.get('MAX_TOKENS', '8192'))
REPEAT_MISS = os.environ.get('REPEAT_MISS', '1') == '1'
TIMEOUT_S = int(os.environ.get('TIMEOUT_S', '7200'))
NAME = os.environ.get('NAME', 'recall')

# (depth from the end in tokens, the user's sentence, the marker word the answer must hold)
FACTS = [
    (48000, "By the way, I have a cat called Pumpernickel.", "pumpernickel"),
    (32000, "I grew up in Dundee, and I still go back there a few times a year.", "dundee"),
    (16000, "I finally washed my car today, an old green Volvo.", "volvo"),
    (4000, "Work is hectic: I am a lighthouse keeper, and the storm season has started.", "lighthouse"),
    (1000, "Tonight I am watching my favourite film again, Metropolis.", "metropolis"),
    (300, "I never drink coffee, only hibiscus tea.", "hibiscus"),
]
ACKS = ["Good to know, thanks for telling me.", "Oh, nice! I will remember that.",
        "Interesting, thanks for sharing.", "Noted. That sounds lovely.",
        "Thanks, that is nice to hear.", "Got it, good to know."]
FILLER = [
    ("How are you today? Anything new?", "All good, thanks! Same routine as usual, but it is nice to chat."),
    ("I watched a series last night, an ordinary detective story.", "Detective stories are a classic. Did you like the plot?"),
    ("I went for a walk in the park; the apple trees have finished flowering.", "Lovely. Spring always lifts the mood."),
    ("I am thinking of doing a big clean-up at the weekend.", "Good plan, resting afterwards feels even better."),
    ("A friend recommended a new bookshop to me.", "Oh, worth a visit. Did you find anything there?"),
    ("I tried cooking a new dish and it came out quite well.", "Well done! What did you make, if you do not mind me asking?"),
    ("I listened to a podcast about travelling.", "Nice. Do you dream of going somewhere?"),
    ("Lots of small chores have piled up since Monday.", "That happens; the trick is to clear them a bit at a time."),
    ("I tried a new route to work and it was quicker.", "Handy, saving time on the commute is always welcome."),
    ("Tonight I plan to rest and do nothing at all.", "Well deserved. Sometimes that is the best plan."),
    ("The weather keeps changing every hour.", "Typical for this time of year. Take an umbrella just in case."),
    ("I rearranged the furniture in the living room.", "A fresh layout can make a room feel new. Do you like it?"),
    ("My neighbour is learning to play the piano.", "That can be charming or tiring, depending on the hour."),
    ("I bought some new plants for the balcony.", "Great idea, a bit of green makes a big difference."),
    ("I finally sorted out my old photos.", "That is satisfying. Did you find any nice memories?"),
    ("I am trying to go to bed earlier this week.", "Good luck! Sleep makes everything easier."),
]
QUESTION = ("Think back over our whole conversation: what have I told you about myself? "
            "List every fact as a short list, without comments.")


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        return json.load(r)


def ntok(text, special=True):
    return len(post('/tokenize', {'content': text, 'add_special': special, 'parse_special': True})['tokens'])


def render(msgs):
    return post('/apply-template', {'messages': msgs})['prompt']


def user(t):
    return {'role': 'user', 'content': t}


def assistant(t):
    return {'role': 'assistant', 'content': t}


def check_markers():
    other = ' '.join([u + ' ' + a for u, a in FILLER] + ACKS + [QUESTION]).lower()
    for _, sentence, marker in FACTS:
        assert marker in sentence.lower(), marker
        assert marker not in other, f'marker {marker} also outside its fact'
        assert sum(marker in s.lower() for _, s, _ in FACTS) == 1, f'marker {marker} in two facts'


def costs():
    """Token costs from the server's template: the question's text to the end of the prompt (base), what the template
    puts before the first message (prefix, e.g. a default system block), and every exchange (user line + assistant
    line, with their markup), measured as rendered before the question."""
    text = render([user(QUESTION)])
    n = ntok(text)
    base = n - ntok(text[:text.index(QUESTION)])
    def pair(u, a):
        return ntok(render([user(u), assistant(a), user(QUESTION)])) - n
    filler = [pair(u, a) for u, a in FILLER]
    facts = [pair(s, ACKS[i % len(ACKS)]) for i, (_, s, _) in enumerate(FACTS)]
    return base, n - base, filler, facts


def build(total, base, prefix, filler, facts):
    """The messages of a prompt of about `total` tokens, built backward from the question. A depth counts from a
    line's first token to the end of the prompt, as measure() does."""
    rng = random.Random(SEED)
    order = sorted(range(len(FACTS)), key=lambda i: FACTS[i][0])  # the shallowest fact first
    half = sum(filler) / len(filler) / 2
    rev, depth, k, prev = [], base, 0, None  # rev: exchanges from the end; depth: tokens from here to the end
    while True:
        if k < len(order) and depth + facts[order[k]] >= FACTS[order[k]][0] - half:
            f = order[k]
            rev.append((FACTS[f][1], ACKS[f % len(ACKS)]))
            depth += facts[f]
            k += 1
            continue
        if prev is None:
            j = rng.randrange(len(FILLER))
        else:  # never the same exchange twice in a row
            j = rng.randrange(len(FILLER) - 1)
            j += j >= prev
        if prefix + depth + filler[j] > total:
            for f in order[k:]:  # facts deeper than the prompt: at its start, the deepest first
                rev.append((FACTS[f][1], ACKS[f % len(ACKS)]))
            break
        rev.append(FILLER[j])
        depth += filler[j]
        prev = j
    msgs = []
    for u, a in reversed(rev):
        msgs += [user(u), assistant(a)]
    return msgs + [user(QUESTION)]


def measure(msgs):
    """The prompt's text, its length and each fact's depth (its first token to the end), by the server's count."""
    text = render(msgs)
    n = ntok(text)
    depths = []
    for _, sentence, _ in FACTS:
        i = text.index(sentence)
        depths.append(n - ntok(text[:i]))
    return text, n, depths


def first_line(s):
    for line in (s or '').splitlines():
        if line.strip():
            return line.strip()
    return ''


def score(content):
    low = (content or '').lower()
    return [m for _, _, m in FACTS if m in low]


COLS = ['name', 'try', 'target', 'prompt_tokens', 'measured_tokens', 'depths', 'found', 'n_found', 'pass',
        'in_reasoning', 'finish_reason', 'completion_tokens', 'prompt_s', 'prompt_tps', 'gen_tps', 'wall_s',
        'sha256', 'first_line']


def run(targets):
    if not OUT:
        sys.exit('OUT=<results folder>')
    os.makedirs(os.path.join(OUT, 'prompts'), exist_ok=True)
    check_markers()
    base, prefix, filler, facts = costs()
    print(f'costs: question {base}, before the first message {prefix}, small talk {min(filler)}-{max(filler)}, '
          f'facts {facts}', flush=True)
    tsv = os.path.join(OUT, 'results.tsv')
    if not os.path.exists(tsv):
        with open(tsv, 'w') as f:
            f.write('\t'.join(COLS) + '\n')
    for target in targets:
        name = f'{NAME}-{target}'
        msgs = build(target, base, prefix, filler, facts)
        text, n, depths = measure(msgs)
        sha = hashlib.sha256(text.encode()).hexdigest()
        with gzip.open(os.path.join(OUT, 'prompts', name + '.json.gz'), 'wt') as f:
            json.dump(msgs, f, ensure_ascii=False)
        print(f'{name}: {n} tokens, {len(msgs)} messages, fact depths {depths}, sha256 {sha[:16]}', flush=True)
        max_tokens = min(MAX_TOKENS, CTX - n - 16)
        body = {'messages': msgs, 'max_tokens': max_tokens, 'temperature': 0.0, 'top_k': 1, 'top_p': 1.0,
                'min_p': 0.0, 'seed': SEED, 'cache_prompt': False, 'stream': False}
        for t in (1, 2):
            t0 = time.time()
            reply = post('/v1/chat/completions', body)
            wall = time.time() - t0
            choice = reply['choices'][0]
            msg = choice.get('message', {})
            content, reasoning = msg.get('content') or '', msg.get('reasoning_content') or ''
            found = score(content)
            tim = reply.get('timings', {})
            row = dict(name=name, target=target, measured_tokens=n, depths=','.join(map(str, depths)),
                       found=','.join(found), n_found=len(found), sha256=sha,
                       prompt_tokens=reply.get('usage', {}).get('prompt_tokens', ''),
                       completion_tokens=reply.get('usage', {}).get('completion_tokens', ''),
                       finish_reason=choice.get('finish_reason', ''),
                       in_reasoning=','.join(score(reasoning)),
                       prompt_s=round(tim.get('prompt_ms', 0) / 1000, 1),
                       prompt_tps=round(tim.get('prompt_per_second', 0), 2),
                       gen_tps=round(tim.get('predicted_per_second', 0), 2), wall_s=round(wall, 1),
                       first_line=first_line(content).replace('\t', ' ')[:200])
            row['try'] = t
            row['pass'] = int(len(found) == len(FACTS))
            with open(os.path.join(OUT, f'{name}-{t}.json'), 'w') as f:
                json.dump({'row': row, 'request': {k: v for k, v in body.items() if k != 'messages'},
                           'reply': reply}, f, ensure_ascii=False, indent=1)
            with open(tsv, 'a') as f:
                f.write('\t'.join(str(row[c]) for c in COLS) + '\n')
            print(f'{name} try {t}: {len(found)}/{len(FACTS)} [{row["found"]}], {row["finish_reason"]}, '
                  f'{row["completion_tokens"]} tokens, prompt {row["prompt_tokens"]} at {row["prompt_tps"]} tok/s, '
                  f'{wall:.0f} s; first line: {row["first_line"][:100]}', flush=True)
            if row['pass'] and t == 1:
                break
            if t == 2:
                print(f'{name} was sent once more after a miss: stopping here', flush=True)
                return 1
            if not REPEAT_MISS:
                break
    return 0


def table():
    rows = []
    with open(os.path.join(OUT, 'results.tsv')) as f:
        head = f.readline().rstrip('\n').split('\t')
        for line in f:
            rows.append(dict(zip(head, line.rstrip('\n').split('\t'))))
    markers = [m for _, _, m in FACTS]
    print('| Prompt | Try | Tokens (server) | Fact depths, tokens from the end | '
          + ' | '.join(f'{m} ({d:,})' for d, _, m in FACTS)
          + ' | Found | Finish | Reply tokens | Prompt tok/s | Reply\'s first line |')
    print('|---|---:|---:|---|' + '---|' * len(FACTS) + '---:|---|---:|---:|---|')
    for r in rows:
        found = r['found'].split(',') if r['found'] else []
        depths = ', '.join(f'{int(d):,}' for d in r['depths'].split(','))
        cells = ['yes' if m in found else '**no**' for m in markers]
        line = r['first_line'].replace('|', '\\|') or '(no answer: the reply has reasoning only)'
        print(f"| {r['name']} | {r['try']} | {int(r['prompt_tokens']):,} | {depths} | " + ' | '.join(cells)
              + f" | {r['n_found']}/{len(FACTS)} | {r['finish_reason']} | {r['completion_tokens']} | "
              f"{r['prompt_tps']} | {line} |")


def show(target):
    base, prefix, filler, facts = costs()
    msgs = build(target, base, prefix, filler, facts)
    text, n, depths = measure(msgs)
    for m in msgs:
        print(f"{m['role']}: {m['content']}")
    print(f'# {n} tokens, fact depths {depths}, sha256 {hashlib.sha256(text.encode()).hexdigest()}')


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('run', 'table', 'show'):
        sys.exit(__doc__)
    if sys.argv[1] == 'run':
        sys.exit(run([int(a) for a in sys.argv[2:]]))
    if sys.argv[1] == 'table':
        table()
    else:
        show(int(sys.argv[2]))
