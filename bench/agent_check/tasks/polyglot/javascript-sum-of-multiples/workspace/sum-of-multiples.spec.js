import { describe, expect, test, xtest } from '@jest/globals';
import { sum } from './sum-of-multiples';

describe('Sum Of Multiples', () => {
  test('no multiples within limit', () => {
    expect(sum([3, 5], 1)).toEqual(0);
  });

  test('more than one factor with multiples within limit', () => {
    expect(sum([3, 5], 10)).toEqual(23);
  });

  test('three factors', () => {
    expect(sum([7, 13, 17], 20)).toEqual(51);
  });

  test('one factor is a multiple of another', () => {
    expect(sum([5, 25], 51)).toEqual(275);
  });

  test('no factors means an empty sum', () => {
    expect(sum([], 10000)).toEqual(0);
  });

  test('solutions using include-exclude must extend to cardinality greater than 3', () => {
    expect(sum([2, 3, 5, 7, 11], 10000)).toEqual(39614537);
  });
});
