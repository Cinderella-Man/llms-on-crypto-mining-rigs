#!/usr/bin/env python3
"""Recorded agent sessions as plain text for llama-perplexity (FN93c, Q1487; also FN95, Q1493).

Usage: agent_text.py <proxy.jsonl.bodies.gz> [TAG_REGEX] > text.txt

Reads the agent check's recorded requests (one JSON line per request: id, body, reply) and the proxy log beside them
(the same name without .bodies.gz, which gives each request's tag). Like bench/deep_lane/deep_client.py, a trace (the
tag without its turn number) is one session and its last request holds the whole session; requests whose tag does not
match TAG_REGEX are skipped. Each session's messages are written in order as plain text: a line with the role, then
the content (for the assistant also its reasoning and each tool call as name and arguments), a blank line between
messages and two between sessions. No chat template: the text is read as text. Prints the sessions and characters to
stderr.
"""
import gzip
import json
import re
import sys


def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # [{type: text, text: ...}, ...]
        return '\n'.join(p.get('text') or '' for p in content if isinstance(p, dict))
    return ''


def render(messages):
    out = []
    for m in messages:
        parts = [text_of(m.get('reasoning_content')), text_of(m.get('content'))]
        for c in m.get('tool_calls') or []:
            f = c.get('function') or {}
            parts.append(f"{f.get('name', '')}({f.get('arguments', '')})")
        body = '\n'.join(p for p in parts if p)
        out.append(f"{m.get('role', '')}:\n{body}")
    return '\n\n'.join(out)


def main():
    path = sys.argv[1]
    tag_re = re.compile(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None
    tags = {}
    with open(path[:-len('.bodies.gz')]) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                tags[r.get('id')] = r.get('tag') or ''
    sessions, order = {}, []
    with gzip.open(path, 'rt') as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            body = d.get('body')
            tag = tags.get(d.get('id'), '')
            if not isinstance(body, dict) or not body.get('messages') or (tag_re and not tag_re.search(tag)):
                continue
            trace = re.sub(r'/\d+$', '', tag)
            if trace not in sessions:
                order.append(trace)
            sessions[trace] = body['messages']  # the trace's last request holds the whole session
    text = '\n\n\n'.join(render(sessions[t]) for t in order) + '\n'
    sys.stdout.write(text)
    print(f'agent_text: {len(order)} sessions, {len(text)} characters', file=sys.stderr)


if __name__ == '__main__':
    main()
