import { describe, expect, test, xtest } from '@jest/globals';
import { all, allSettled, any, promisify, race } from './promises';

describe('promises', () => {
  const failedCallback = new Error('Failed callback');

  const createCallbackFn = (speed) => (value, callback) =>
    setTimeout(() => callback(null, value), speed);
  const createFailedCallback = (speed) => (_, callback) =>
    setTimeout(() => callback(failedCallback), speed);

  const slowestCallbackFn = createCallbackFn(20);
  const slowerCallbackFn = createCallbackFn(10);
  const fastCallbackFn = createCallbackFn(0);
  const failedCallbackFn = createFailedCallback(10);

  describe('promisify', () => {
    test('returns a function', () => {
      expect(typeof promisify(fastCallbackFn)).toBe('function');
    });

    test("promisified function rejects a callback's error", () => {
      const failedPromise = promisify(failedCallbackFn);
      return expect(failedPromise(null)).rejects.toEqual(failedCallback);
    });
  });

  describe('all', () => {
    const [slowestPromise, slowerPromise, fastPromise, failedPromise] = [
      slowestCallbackFn,
      slowerCallbackFn,
      fastCallbackFn,
      failedCallbackFn,
    ].map((fn) => promisify(fn));

    test('returns a Promise', () => {
      expect(all([])).toBeInstanceOf(Promise);
    });

    test('resolved values appear in the order they are passed in', () => {
      const FIRST = 'FIRST';
      const SECOND = 'SECOND';
      const THIRD = 'THIRD';
      const result = all([
        slowestPromise(FIRST),
        slowerPromise(SECOND),
        fastPromise(THIRD),
      ]);
      return expect(result).resolves.toEqual([FIRST, SECOND, THIRD]);
    });
  });

  describe('allSettled', () => {
    const [slowestPromise, slowerPromise, fastPromise, failedPromise] = [
      slowestCallbackFn,
      slowerCallbackFn,
      fastCallbackFn,
      failedCallbackFn,
    ].map((fn) => promisify(fn));

    test('returns a Promise', () => {
      expect(allSettled([])).toBeInstanceOf(Promise);
    });

    test('resolved values appear in the order they are passed in', () => {
      const FIRST = 'FIRST';
      const SECOND = 'SECOND';
      const THIRD = 'THIRD';
      const result = allSettled([
        slowestPromise(FIRST),
        slowerPromise(SECOND),
        fastPromise(THIRD),
      ]);
      return expect(result).resolves.toEqual([FIRST, SECOND, THIRD]);
    });
  });

  describe('race', () => {
    const [slowestPromise, slowerPromise, fastPromise, failedPromise] = [
      slowestCallbackFn,
      slowerCallbackFn,
      fastCallbackFn,
      failedCallbackFn,
    ].map((fn) => promisify(fn));

    test('returns a Promise', () => {
      expect(race([])).toBeInstanceOf(Promise);
    });

    test('resolves with value of the fastest successful promise', () => {
      const FAST = 'FAST';
      return expect(
        race([
          slowestPromise('SLOWEST'),
          slowerPromise('SLOWER'),
          fastPromise(FAST),
        ]),
      ).resolves.toEqual(FAST);
    });
  });

  describe('any', () => {
    const [slowestPromise, slowerPromise, fastPromise, failedPromise] = [
      slowestCallbackFn,
      slowerCallbackFn,
      fastCallbackFn,
      failedCallbackFn,
    ].map((fn) => promisify(fn));

    test('returns a Promise', () => {
      expect(any([]).catch(() => null)).toBeInstanceOf(Promise);
    });

    test('resolves with value of fastest successful promise', () => {
      const FAST = 'FAST';
      return expect(
        any([
          slowestPromise('SLOWEST'),
          slowerPromise('SLOWER'),
          fastPromise(FAST),
        ]),
      ).resolves.toEqual(FAST);
    });

    test('rejects with array of errors if all promises fail', () => {
      return expect(
        any([failedPromise(null), failedPromise(null)]),
      ).rejects.toEqual([failedCallback, failedCallback]);
    });
  });
});
