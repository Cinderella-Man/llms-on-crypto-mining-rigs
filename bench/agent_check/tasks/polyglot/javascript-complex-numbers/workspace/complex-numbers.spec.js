import { describe, expect, test, xtest } from '@jest/globals';
import { ComplexNumber } from './complex-numbers';

describe('Complex numbers', () => {
  describe('Real part', () => {
    test('Real part of a purely real number', () => {
      const expected = 1;
      const actual = new ComplexNumber(1, 0).real;

      expect(actual).toEqual(expected);
    });
  });

  describe('Imaginary part', () => {
    test('Imaginary part of a purely real number', () => {
      const expected = 0;
      const actual = new ComplexNumber(1, 0).imag;

      expect(actual).toEqual(expected);
    });
  });

  test('Imaginary unit', () => {
    const expected = new ComplexNumber(-1, 0);
    const actual = new ComplexNumber(0, 1).mul(new ComplexNumber(0, 1));

    expect(actual).toEqual(expected);
  });

  describe('Arithmetic', () => {
    test('Add purely real numbers', () => {
      const expected = new ComplexNumber(3, 0);
      const actual = new ComplexNumber(1, 0).add(new ComplexNumber(2, 0));

      expect(actual).toEqual(expected);
    });

    test('Subtract purely real numbers', () => {
      const expected = new ComplexNumber(-1, 0);
      const actual = new ComplexNumber(1, 0).sub(new ComplexNumber(2, 0));

      expect(actual).toEqual(expected);
    });

    test('Multiply purely real numbers', () => {
      const expected = new ComplexNumber(2, 0);
      const actual = new ComplexNumber(1, 0).mul(new ComplexNumber(2, 0));

      expect(actual).toEqual(expected);
    });

    test('Divide purely real numbers', () => {
      const expected = new ComplexNumber(0.5, 0);
      const actual = new ComplexNumber(1, 0).div(new ComplexNumber(2, 0));

      expect(actual).toEqual(expected);
    });
  });

  describe('Absolute value', () => {
    test('Absolute value of a positive purely real number', () => {
      const expected = 5;
      const actual = new ComplexNumber(5, 0).abs;

      expect(actual).toEqual(expected);
    });

    test('Absolute value of a purely imaginary number with negative imaginary part', () => {
      const expected = 5;
      const actual = new ComplexNumber(0, -5).abs;

      expect(actual).toEqual(expected);
    });
  });

  describe('Complex conjugate', () => {
    test('Conjugate a purely real number', () => {
      const expected = new ComplexNumber(5, 0);
      const actual = new ComplexNumber(5, 0).conj;

      expect(actual).toEqual(expected);
    });
  });

  describe('Complex exponential function', () => {
    test("Euler's identity/formula", () => {
      const expected = new ComplexNumber(-1, 0);
      const actual = new ComplexNumber(0, Math.PI).exp;

      expect(actual.real).toBeCloseTo(expected.real);
      expect(actual.imag).toBeCloseTo(expected.imag);
    });

    test('Exponential of a number with real and imaginary part', () => {
      const expected = new ComplexNumber(-2, 0);
      const actual = new ComplexNumber(Math.LN2, Math.PI).exp;

      expect(actual.real).toBeCloseTo(expected.real);
      expect(actual.imag).toBeCloseTo(expected.imag);
    });
  });

  describe('Operations between real numbers and complex numbers', () => {
    test('Add real number to complex number', () => {
      const expected = new ComplexNumber(6, 2);
      const actual = new ComplexNumber(1, 2).add(new ComplexNumber(5, 0));

      expect(actual.real).toBeCloseTo(expected.real);
      expect(actual.imag).toBeCloseTo(expected.imag);
    });

    test('Subtract complex number from real number', () => {
      const expected = new ComplexNumber(-1, -7);
      const actual = new ComplexNumber(4, 0).sub(new ComplexNumber(5, 7));

      expect(actual.real).toBeCloseTo(expected.real);
      expect(actual.imag).toBeCloseTo(expected.imag);
    });

    test('Divide complex number by real number', () => {
      const expected = new ComplexNumber(1, 10);
      const actual = new ComplexNumber(10, 100).div(new ComplexNumber(10, 0));

      expect(actual.real).toBeCloseTo(expected.real);
      expect(actual.imag).toBeCloseTo(expected.imag);
    });
  });
});
