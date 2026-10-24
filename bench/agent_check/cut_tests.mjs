// cut_tests.mjs SPEC VISIBLE_OUT HIDDEN_OUT: split an Exercism JavaScript spec into the visible third and the full
// hidden file. Both get every skipped test (xtest, xit, xdescribe) switched on. The visible file keeps, inside each
// innermost describe block, the tests at positions 0, 3, 6, ... (at least one per block); the others are cut with
// their leading comments. Prints {"total": n, "visible": k, "groups": g} as JSON.
// Needs @babel/parser, found through AC_NODE_MODULES (the shared node_modules of env/package.json).
import { readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';

const require = createRequire(path.join(process.env.AC_NODE_MODULES ?? './node_modules', 'x.js'));
const { parse } = require('@babel/parser');

const [spec, visibleOut, hiddenOut] = process.argv.slice(2);
const src = readFileSync(spec, 'utf8');
const ast = parse(src, { sourceType: 'module', attachComment: true });

const TESTS = new Set(['test', 'xtest', 'it', 'xit']);
const RENAME = { xtest: 'test', xit: 'it', xdescribe: 'describe' };
const renames = [];          // [start, end, text]
const groups = new Map();    // describe call start (or -1) -> [ExpressionStatement]

function calleeName(call) {
  return call.callee.type === 'Identifier' ? call.callee.name : null;
}

function walk(node, group) {
  if (!node || typeof node.type !== 'string') return;
  if (node.type === 'CallExpression') {
    const name = calleeName(node);
    if (name && RENAME[name]) renames.push([node.callee.start, node.callee.end, RENAME[name]]);
    if (name === 'describe' || name === 'xdescribe') group = node.start;
  }
  if (node.type === 'ExpressionStatement' && node.expression.type === 'CallExpression'
      && TESTS.has(calleeName(node.expression))) {
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(node);
  }
  for (const key of Object.keys(node)) {
    if (key === 'leadingComments' || key === 'trailingComments' || key === 'innerComments') continue;
    const v = node[key];
    if (Array.isArray(v)) v.forEach((c) => walk(c, group));
    else if (v && typeof v.type === 'string') walk(v, group);
  }
}
walk(ast.program, -1);

function lineStart(pos) {
  let p = pos;
  while (p > 0 && src[p - 1] !== '\n') p--;
  return p;
}

const cuts = [];             // [start, end]
let total = 0;
let visible = 0;
for (const stmts of groups.values()) {
  stmts.forEach((st, j) => {
    total++;
    if (j % 3 === 0) { visible++; return; }
    let start = st.leadingComments?.length ? st.leadingComments[0].start : st.start;
    start = lineStart(start);
    // also drop one blank line in front, so the cut leaves the spacing of the original
    if (start >= 2 && src[start - 1] === '\n' && src[start - 2] === '\n') start -= 1;
    let end = st.end;
    if (src[end] === '\n') end += 1;
    cuts.push([start, end]);
  });
}

function apply(edits) {
  // edits: [start, end, text], non-overlapping; applied from the end
  let out = src;
  for (const [s, e, t] of [...edits].sort((a, b) => b[0] - a[0])) out = out.slice(0, s) + t + out.slice(e);
  return out;
}

const inCut = (pos) => cuts.some(([s, e]) => pos >= s && pos < e);
writeFileSync(hiddenOut, apply(renames));
writeFileSync(visibleOut, apply([...cuts.map(([s, e]) => [s, e, '']), ...renames.filter(([s]) => !inCut(s))]));
console.log(JSON.stringify({ total, visible, groups: groups.size }));
