import { describe, expect, test, xtest } from '@jest/globals';
import { parallelLetterFrequency } from './parallel-letter-frequency';

describe('ParallelLetterFrequency', () => {
  test('no texts', async () => {
    const expected = {};
    const actual = parallelLetterFrequency([]);
    await expect(actual).resolves.toEqual(expected);
  });

  test('two texts with one letter', async () => {
    const texts = ['e', 'f'];
    const expected = {
      e: 1,
      f: 1,
    };
    const actual = parallelLetterFrequency(texts);
    await expect(actual).resolves.toEqual(expected);
  });

  test('ignore whitespace', async () => {
    const texts = ['   ', '\t', '\r\n'];
    const expected = {};
    const actual = parallelLetterFrequency(texts);
    await expect(actual).resolves.toEqual(expected);
  });

  test('Unicode letters', async () => {
    const texts = ['本', 'φ', 'ほ', 'ø'];
    const expected = {
      本: 1,
      φ: 1,
      ほ: 1,
      ø: 1,
    };
    const actual = parallelLetterFrequency(texts);
    await expect(actual).resolves.toEqual(expected);
  });

  test('many small texts', async () => {
    const texts = Array(50).fill('abbccc');
    const expected = {
      a: 50,
      b: 100,
      c: 150,
    };
    const actual = parallelLetterFrequency(texts);
    await expect(actual).resolves.toEqual(expected);
  });
});
