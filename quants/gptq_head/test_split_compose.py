#!/usr/bin/env python3
# Checks for split_compose.py on small synthetic GGUFs: base + shard hold every source tensor once, a tensor named by
# several donors comes from the first one, verify accepts that pair and rejects it against the donors in another order,
# and a donor given as path@regex contributes only the tensors the regex matches.
#
#   test_split_compose.py
#
# Needs numpy and gguf-py (PYTHONPATH=<llama.cpp>/gguf-py).
import os, subprocess, sys, tempfile
import numpy as np
import gguf

HERE = os.path.dirname(os.path.abspath(__file__))
fails = []


def check(ok, what):
    print(('ok    ' if ok else 'FAIL  ') + what)
    if not ok:
        fails.append(what)


def write(path, tensors, kv=True):
    w = gguf.GGUFWriter(path, arch='llama')
    if kv:
        w.add_uint32('llama.block_count', 2)
        w.add_string('general.name', 'synthetic')
    for n, a in tensors.items():
        w.add_tensor(n, a)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file()
    w.close()


def run(*args):
    return subprocess.run([sys.executable, os.path.join(HERE, 'split_compose.py'), *args], capture_output=True, text=True)


rng = np.random.default_rng(0)
names = [f'blk.{l}.{k}.weight' for l in range(2) for k in ('attn_q', 'attn_k', 'ffn_up', 'ffn_up_exps')]
src = {n: rng.standard_normal((8, 32)).astype(np.float32) for n in names}
d1 = {'blk.0.attn_q.weight': rng.standard_normal((8, 32)).astype(np.float32)}
d2 = {'blk.0.attn_q.weight': rng.standard_normal((8, 32)).astype(np.float32),
      'blk.1.attn_k.weight': rng.standard_normal((8, 32)).astype(np.float32)}
rx = r'^blk\.\d+\.(attn_(q|k)|ffn_up)\.weight$'   # every always-read matrix here; leaves the experts in the base

with tempfile.TemporaryDirectory() as t:
    for f, d in (('src', src), ('d1', d1), ('d2', d2)):
        write(f'{t}/{f}.gguf', d, kv=f == 'src')
    r = run('base', f'{t}/src.gguf', f'{t}/base.gguf', rx)
    check(r.returncode == 0, 'base: ' + r.stdout.strip())
    bt = {x.name for x in gguf.GGUFReader(f'{t}/base.gguf').tensors}
    check(bt == {n for n in names if 'exps' in n}, f'base holds only the tensors outside the regex ({sorted(bt)})')
    r = run('shard', f'{t}/base.gguf', f'{t}/src.gguf', f'{t}/v', 'var', rx, f'{t}/d1.gguf,{t}/d2.gguf')
    check(r.returncode == 0, 'shard with two donors: ' + r.stdout.strip())
    p1, p2 = f'{t}/v/var-00001-of-00002.gguf', f'{t}/v/var-00002-of-00002.gguf'
    check(os.path.islink(p1) and os.path.realpath(p1) == os.path.realpath(f'{t}/base.gguf'), 'first file links to the base')
    st = {x.name: np.asarray(x.data) for x in gguf.GGUFReader(p2).tensors}
    check(np.array_equal(st['blk.0.attn_q.weight'], d1['blk.0.attn_q.weight']), 'a tensor both donors have comes from the first')
    check(np.array_equal(st['blk.1.attn_k.weight'], d2['blk.1.attn_k.weight']), 'a tensor only the second donor has comes from it')
    check(np.array_equal(st['blk.1.ffn_up.weight'], src['blk.1.ffn_up.weight']), 'a tensor no donor has comes from the source')
    r2 = gguf.GGUFReader(p2)
    meta = {f.name: f.contents() for f in r2.fields.values() if f.name.startswith('split.')}
    check(meta == {'split.no': 1, 'split.count': 2, 'split.tensors.count': len(names)}, f'split metadata {meta}')
    meta1 = {f.name: f.contents() for f in gguf.GGUFReader(p1).fields.values() if f.name.startswith('split.')}
    check(meta1 == {'split.no': 0, 'split.count': 2, 'split.tensors.count': len(names)}, f'base split metadata {meta1}')
    r = run('verify', f'{t}/src.gguf', p1, f'{t}/d1.gguf,{t}/d2.gguf')
    check(r.returncode == 0, 'verify, same donor order: ' + (r.stdout.strip() or r.stderr.strip()))
    r = run('verify', f'{t}/src.gguf', p1, f'{t}/d2.gguf,{t}/d1.gguf')
    check(r.returncode != 0, 'verify rejects the pair against the donors in the other order: ' + r.stderr.strip())
    r = run('verify', f'{t}/src.gguf', p1)
    check(r.returncode != 0, 'verify rejects the pair without donors: ' + r.stderr.strip())
    # a donor filtered by a regex contributes only the tensors it matches
    r = run('shard', f'{t}/base.gguf', f'{t}/src.gguf', f'{t}/w', 'filt', rx, f'{t}/d2.gguf@attn_k')
    st = {x.name: np.asarray(x.data) for x in gguf.GGUFReader(f'{t}/w/filt-00002-of-00002.gguf').tensors}
    check(r.returncode == 0 and np.array_equal(st['blk.1.attn_k.weight'], d2['blk.1.attn_k.weight'])
          and np.array_equal(st['blk.0.attn_q.weight'], src['blk.0.attn_q.weight']),
          'donor@regex: the matching tensor from the donor, the filtered-out one from the source')
    r = run('verify', f'{t}/src.gguf', f'{t}/w/filt-00001-of-00002.gguf', f'{t}/d2.gguf@attn_k')
    check(r.returncode == 0, 'verify with the same filtered donor: ' + (r.stdout.strip() or r.stderr.strip()))

print('all passed' if not fails else f'{len(fails)} FAILED')
sys.exit(1 if fails else 0)
