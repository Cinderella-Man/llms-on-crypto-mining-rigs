import { beforeEach, describe, expect, test, xtest } from '@jest/globals';
import { Forth } from './forth';

describe('Forth', () => {
  let forth;

  beforeEach(() => {
    forth = new Forth();
  });

  describe('parsing and numbers', () => {
    test('numbers just get pushed onto the stack', () => {
      forth.evaluate('1 2 3 4 5');
      expect(forth.stack).toEqual([1, 2, 3, 4, 5]);
    });
  });

  describe('addition', () => {
    test('can add two numbers', () => {
      forth.evaluate('1 2 +');
      expect(forth.stack).toEqual([3]);
    });

    test('more than two values on the stack', () => {
      forth.evaluate('1 2 3 +');
      expect(forth.stack).toEqual([1, 5]);
    });
  });

  describe('subtraction', () => {
    test('can subtract two numbers', () => {
      forth.evaluate('3 4 -');
      expect(forth.stack).toEqual([-1]);
    });

    test('more than two values on the stack', () => {
      forth.evaluate('1 12 3 -');
      expect(forth.stack).toEqual([1, 9]);
    });
  });

  describe('multiplication', () => {
    test('can multiply two numbers', () => {
      forth.evaluate('2 4 *');
      expect(forth.stack).toEqual([8]);
    });

    test('more than two values on the stack', () => {
      forth.evaluate('1 2 3 *');
      expect(forth.stack).toEqual([1, 6]);
    });
  });

  describe('division', () => {
    test('can divide two numbers', () => {
      forth.evaluate('12 3 /');
      expect(forth.stack).toEqual([4]);
    });

    test('errors if there is nothing on the stack', () => {
      expect(() => {
        forth.evaluate('/');
      }).toThrow(new Error('Stack empty'));
    });
  });

  describe('combined arithmetic', () => {
    test('addition and subtraction', () => {
      forth.evaluate('1 2 + 4 -');
      expect(forth.stack).toEqual([-1]);
    });

    test('addition and multiplication', () => {
      forth.evaluate('1 3 4 + *');
      expect(forth.stack).toEqual([7]);
    });
  });

  describe('dup', () => {
    test('copies a value on the stack', () => {
      forth.evaluate('1 dup');
      expect(forth.stack).toEqual([1, 1]);
    });
  });

  describe('drop', () => {
    test('removes the top value on the stack if it is the only one', () => {
      forth.evaluate('1 drop');
      expect(forth.stack).toEqual([]);
    });
  });

  describe('swap', () => {
    test('swaps the top two values on the stack if they are the only ones', () => {
      forth.evaluate('1 2 swap');
      expect(forth.stack).toEqual([2, 1]);
    });

    test('errors if there is only one value on the stack', () => {
      expect(() => {
        forth.evaluate('1 swap');
      }).toThrow(new Error('Only one value on the stack'));
    });
  });

  describe('over', () => {
    test('copies the second element if there are only two', () => {
      forth.evaluate('1 2 over');
      expect(forth.stack).toEqual([1, 2, 1]);
    });

    test('errors if there is only one value on the stack', () => {
      expect(() => {
        forth.evaluate('1 over');
      }).toThrow(new Error('Only one value on the stack'));
    });
  });

  describe('user-defined words', () => {
    test('can consist of built-in words', () => {
      forth.evaluate(': dup-twice dup dup ;');
      forth.evaluate('1 dup-twice');
      expect(forth.stack).toEqual([1, 1, 1]);
    });

    test('can override built-in words', () => {
      forth.evaluate(': swap dup ;');
      forth.evaluate('1 swap');
      expect(forth.stack).toEqual([1, 1]);
    });

    test('can define word that uses word with the same name', () => {
      forth.evaluate(': foo 10 ;');
      forth.evaluate(': foo foo 1 + ;');
      forth.evaluate('foo');
      expect(forth.stack).toEqual([11]);
    });

    test('errors if executing a non-existent word', () => {
      expect(() => {
        forth.evaluate('foo');
      }).toThrow(new Error('Unknown command'));
    });
  });

  describe('case-insensitivity', () => {
    test('DUP is case-insensitive', () => {
      forth.evaluate('1 DUP Dup dup');
      expect(forth.stack).toEqual([1, 1, 1, 1]);
    });

    test('OVER is case-insensitive', () => {
      forth.evaluate('1 2 OVER Over over');
      expect(forth.stack).toEqual([1, 2, 1, 2, 1]);
    });
  });
});
