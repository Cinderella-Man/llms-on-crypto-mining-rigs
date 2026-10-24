import { describe, expect, test, xtest } from '@jest/globals';
import { Palindromes } from './palindrome-products';

describe('Palindromes', () => {
  test('smallest palindrome from single digit factors', () => {
    const palindromes = Palindromes.generate({ maxFactor: 9, minFactor: 1 });
    const smallest = palindromes.smallest;
    const expected = { value: 1, factors: [[1, 1]] };

    expect(smallest.value).toEqual(expected.value);
    expect(sortFactors(smallest.factors)).toEqual(expected.factors);
  });

  test('largest palindrome from double digit factors', () => {
    const palindromes = Palindromes.generate({ maxFactor: 99, minFactor: 10 });
    const largest = palindromes.largest;
    const expected = { value: 9009, factors: [[91, 99]] };

    expect(largest.value).toEqual(expected.value);
    expect(sortFactors(largest.factors)).toEqual(expected.factors);
  });

  test('smallest palindrome from four digit factors', () => {
    const palindromes = Palindromes.generate({
      maxFactor: 9999,
      minFactor: 1000,
    });
    const smallest = palindromes.smallest;
    const expected = { value: 1002001, factors: [[1001, 1001]] };

    expect(smallest.value).toEqual(expected.value);
    expect(sortFactors(smallest.factors)).toEqual(expected.factors);
  });

  // This test doesn't run on our online test runner because it will time-out
  // with most implementations. It's up to you to test your solution locally.
  test.skip(
    'largest palindrome from four digit factors',
    () => {
      const palindromes = Palindromes.generate({
        maxFactor: 9999,
        minFactor: 1000,
      });
      const largest = palindromes.largest;
      const expected = { value: 99000099, factors: [[9901, 9999]] };

      expect(largest.value).toEqual(expected.value);
      expect(sortFactors(largest.factors)).toEqual(expected.factors);
    },
    20 * 1000,
  );

  test('error for smallest if min is more than max', () => {
    expect(() => {
      const palindromes = Palindromes.generate({
        maxFactor: 1,
        minFactor: 10000,
      });
      palindromes.smallest;
    }).toThrow(new Error('min must be <= max'));
  });
});

test('smallest product does not use the smallest factor', () => {
  const palindromes = Palindromes.generate({
    maxFactor: 4000,
    minFactor: 3215,
  });
  const smallest = palindromes.smallest;
  const expected = { value: 10988901, factors: [[3297, 3333]] };

  expect(smallest.value).toEqual(expected.value);
  expect(sortFactors(smallest.factors)).toEqual(expected.factors);
});

function sortFactors(factors) {
  return factors.map((f) => f.sort()).sort();
}
