import { describe, expect, test, xtest } from '@jest/globals';
import { cost } from './book-store';

describe('Book Store', () => {
  describe('Creating a basket', () => {
    test('only a single book', () => {
      const basket = [1];
      expect(cost(basket)).toEqual(800);
    });

    test('two different books', () => {
      const basket = [1, 2];
      expect(cost(basket)).toEqual(1520);
    });

    test('five different books', () => {
      const basket = [1, 2, 3, 4, 5];
      expect(cost(basket)).toEqual(3000);
    });

    test('group of four plus group of two is cheaper than two groups of three', () => {
      const basket = [1, 1, 2, 2, 3, 4];
      expect(cost(basket)).toEqual(4080);
    });

    test('three copies of first book and 2 each of remaining', () => {
      const basket = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 1];
      expect(cost(basket)).toEqual(6800);
    });

    test('two groups of four and a group of five', () => {
      const basket = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5];
      expect(cost(basket)).toEqual(8120);
    });
  });
});
