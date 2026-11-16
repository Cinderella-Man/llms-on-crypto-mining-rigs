"""per_task.py ATTEMPTS ARM...: per arm the attempts' sums (decode, prefill, prompt, output, turns) and per task the
result, wall seconds, turns, decode seconds, new prompt tokens, output tokens, tool calls and failed calls."""
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1], newline='')))
for arm in sys.argv[2:]:
    xs = [x for x in rows if x['tag'].startswith(arm + '/')]
    f = lambda k: sum(float(x[k] or 0) for x in xs)
    print(arm, len(xs), 'passed', sum(int(x['pass']) for x in xs), 'wall_s %.0f' % f('wall_s'), 'decode_s %.0f' % f('decode_s'),
          'prefill_s %.1f' % f('prefill_s'), 'prompt_new %.0f' % f('prompt_new'), 'output %.0f' % f('output_tokens'),
          'turns %d' % f('turns'), 'tool_calls %d' % f('tool_calls'), 'tool_errors %d' % f('tool_errors'))
    for x in sorted(xs, key=lambda x: x['task_id']):
        print('  ', x['task_id'], 'pass' if x['pass'] == '1' else 'FAIL', x['wall_s'], 'turns', x['turns'], 'decode_s', x['decode_s'],
              'prompt_new', x['prompt_new'], 'output', x['output_tokens'], 'calls', x['tool_calls'], 'failed', x['tool_errors'],
              'peak_ctx', x['peak_context'])
