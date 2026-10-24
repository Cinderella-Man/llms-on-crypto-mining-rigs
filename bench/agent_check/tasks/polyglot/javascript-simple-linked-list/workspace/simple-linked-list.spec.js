import { beforeEach, describe, expect, test, xtest } from '@jest/globals';
import { Element, List } from './simple-linked-list';

describe('Element class', () => {
  test('has constructor', () => {
    const element = new Element(1);
    expect(element.value).toEqual(1);
  });
});

describe('List class', () => {
  test('has constructor', () => {
    const list = new List();
    expect(list).toBeDefined();
  });

  test('adding a element increments length', () => {
    const list = new List();
    const element = new Element(1);
    list.add(element);
    expect(list.length).toEqual(1);
  });

  test('adding an Element to an empty list sets the head Element', () => {
    const list = new List();
    const element = new Element(1);
    list.add(element);
    expect(list.head.value).toEqual(1);
  });

  test('can be initialized with an array', () => {
    const list = new List([1, 2, 3]);
    expect(list.length).toEqual(3);
    expect(list.head.value).toEqual(3);
  });
});

describe('Lists with multiple elements', () => {
  let list;
  beforeEach(() => {
    list = new List([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  });
  test('with correct length', () => {
    expect(list.length).toEqual(10);
  });
  test('can convert to an array', () => {
    const oneList = new List([1]);
    expect(oneList.toArray()).toEqual([1]);
  });
  test('can be reversed', () => {
    const twoList = new List([1, 2]);
    expect(twoList.reverse().toArray()).toEqual([1, 2]);
  });
  test('can reverse a reversal', () => {
    expect(list.reverse().reverse().toArray()).toEqual([
      10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
    ]);
  });
});
