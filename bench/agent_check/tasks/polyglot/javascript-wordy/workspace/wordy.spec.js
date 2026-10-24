import { describe, expect, test, xtest } from '@jest/globals';
import { answer } from './wordy';

describe('Wordy', () => {
  test('just a number', () => {
    expect(answer('What is 5?')).toEqual(5);
  });

  test('addition', () => {
    expect(answer('What is 1 plus 1?')).toEqual(2);
  });

  test('more addition', () => {
    expect(answer('What is 53 plus 2?')).toEqual(55);
  });

  test('subtraction', () => {
    expect(answer('What is 4 minus -12?')).toEqual(16);
  });

  test('multiple additions', () => {
    expect(answer('What is 1 plus 1 plus 1?')).toEqual(3);
  });

  test('subtraction then addition', () => {
    expect(answer('What is 17 minus 6 plus 3?')).toEqual(14);
  });

  test('multiple division', () => {
    expect(answer('What is -12 divided by 2 divided by -3?')).toEqual(2);
  });

  test('reject problem missing an operand', () => {
    expect(() => answer('What is 1 plus?')).toThrow(new Error('Syntax error'));
  });

  test('reject two numbers in a row', () => {
    expect(() => answer('What is 1 plus 2 1?')).toThrow(
      new Error('Syntax error'),
    );
  });
});
