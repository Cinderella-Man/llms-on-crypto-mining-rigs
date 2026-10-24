import { describe, expect, test, xtest } from '@jest/globals';
import { combinations } from './killer-sudoku-helper';

describe('Trivial 1-digit cages', () => {
  test('1', () => {
    const inputCage = {
      sum: 1,
      size: 1,
      exclude: [],
    };
    const expected = [[1]];
    const actual = combinations(inputCage);
    expect(actual).toEqual(expected);
  });

  test('4', () => {
    const inputCage = {
      sum: 4,
      size: 1,
      exclude: [],
    };
    const expected = [[4]];
    const actual = combinations(inputCage);
    expect(actual).toEqual(expected);
  });

  test('7', () => {
    const inputCage = {
      sum: 7,
      size: 1,
      exclude: [],
    };
    const expected = [[7]];
    const actual = combinations(inputCage);
    expect(actual).toEqual(expected);
  });
});

describe('Other cages', () => {
  test('Cage with sum 45 contains all digits 1:9', () => {
    const inputCage = {
      sum: 45,
      size: 9,
      exclude: [],
    };
    const expected = [[1, 2, 3, 4, 5, 6, 7, 8, 9]];
    const actual = combinations(inputCage);
    expect(actual).toEqual(expected);
  });

  test('Cage with several combinations that is restricted', () => {
    const inputCage = {
      sum: 10,
      size: 2,
      exclude: [1, 4],
    };
    const expected = [
      [2, 8],
      [3, 7],
    ];
    const actual = combinations(inputCage);
    expect(actual).toEqual(expected);
  });
});
