import { describe, expect, test, xtest } from '@jest/globals';
import { solve } from './alphametics';

describe('Solve the alphametics puzzle', () => {
  test('puzzle with three letters', () => {
    const puzzle = 'I + BB == ILL';
    const expected = {
      I: 1,
      B: 9,
      L: 0,
    };
    expect(solve(puzzle)).toEqual(expected);
  });

  test('puzzle with four letters', () => {
    const puzzle = 'AS + A == MOM';
    const expected = {
      A: 9,
      S: 2,
      M: 1,
      O: 0,
    };
    expect(solve(puzzle)).toEqual(expected);
  });

  test('puzzle with eight letters', () => {
    const puzzle = 'SEND + MORE == MONEY';
    const expected = {
      S: 9,
      E: 5,
      N: 6,
      D: 7,
      M: 1,
      O: 0,
      R: 8,
      Y: 2,
    };
    expect(solve(puzzle)).toEqual(expected);
  });
});
