import { describe, expect, test, xtest } from '@jest/globals';
import { count } from './rectangles';

describe('Rectangles', () => {
  test('no rows', () => {
    const expected = 0;
    const actual = count([]);

    expect(actual).toEqual(expected);
  });

  test('one rectangle', () => {
    const expected = 1;
    const actual = count(['+-+', '| |', '+-+']);

    expect(actual).toEqual(expected);
  });

  test('rectangle of height 1 is counted', () => {
    const expected = 1;
    const actual = count(['+--+', '+--+']);

    expect(actual).toEqual(expected);
  });

  test('only complete rectangles are counted', () => {
    const expected = 1;
    const actual = count(['  +-+', '    |', '+-+-+', '| | -', '+-+-+']);

    expect(actual).toEqual(expected);
  });

  test('large input with many rectangles', () => {
    const expected = 60;
    const actual = count([
      '+---+--+----+',
      '|   +--+----+',
      '+---+--+    |',
      '|   +--+----+',
      '+---+--+--+-+',
      '+---+--+--+-+',
      '+------+  | |',
      '          +-+',
    ]);

    expect(actual).toEqual(expected);
  });
});
