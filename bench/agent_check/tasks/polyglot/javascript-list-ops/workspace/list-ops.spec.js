import { describe, expect, test, xtest } from '@jest/globals';
import { List } from './list-ops';

describe('append entries to a list and return the new list', () => {
  test('empty lists', () => {
    const list1 = new List();
    const list2 = new List();
    expect(list1.append(list2)).toEqual(new List());
  });

  test('non-empty lists', () => {
    const list1 = new List([1, 2]);
    const list2 = new List([2, 3, 4, 5]);
    expect(list1.append(list2).values).toEqual([1, 2, 2, 3, 4, 5]);
  });
});

describe('concat lists and lists of lists into new list', () => {
  test('empty list', () => {
    const list1 = new List();
    const list2 = new List();
    expect(list1.concat(list2).values).toEqual([]);
  });
});

describe('filter list returning only values that satisfy the filter function', () => {
  test('empty list', () => {
    const list1 = new List([]);
    expect(list1.filter((el) => el % 2 === 1).values).toEqual([]);
  });
});

describe('returns the length of a list', () => {
  test('empty list', () => {
    const list1 = new List();
    expect(list1.length()).toEqual(0);
  });
});

describe('returns a list of elements whose values equal the list value transformed by the mapping function', () => {
  test('empty list', () => {
    const list1 = new List();
    expect(list1.map((el) => ++el).values).toEqual([]);
  });
});

describe('folds (reduces) the given list from the left with a function', () => {
  test('empty list', () => {
    const list1 = new List();
    expect(list1.foldl((acc, el) => el * acc, 2)).toEqual(2);
  });
});

describe('folds (reduces) the given list from the right with a function', () => {
  test('empty list', () => {
    const list1 = new List();
    expect(list1.foldr((acc, el) => el * acc, 2)).toEqual(2);
  });
});

describe('reverse the elements of a list', () => {
  test('empty list', () => {
    const list1 = new List();
    expect(list1.reverse().values).toEqual([]);
  });
});
