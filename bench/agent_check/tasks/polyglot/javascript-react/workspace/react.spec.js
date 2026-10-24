import { describe, expect, test, xtest } from '@jest/globals';
import { CallbackCell, ComputeCell, InputCell } from './react';

describe('React module', () => {
  test('accepts input', () => {
    const inputCell = new InputCell(10);
    expect(inputCell.value).toEqual(10);
  });

  test('compute cell takes inputs in correct order', () => {
    const inputCells = [new InputCell(1), new InputCell(2)];

    const computeCell = new ComputeCell(
      inputCells,
      (inputs) => inputs[0].value + inputs[1].value * 10,
    );

    expect(computeCell.value).toEqual(21);
  });

  test('compute cells fire callbacks', () => {
    const inputCell = new InputCell(1);
    const output = new ComputeCell(
      [inputCell],
      (inputs) => inputs[0].value + 1,
    );

    const callback = new CallbackCell((cell) => cell.value);
    output.addCallback(callback);

    inputCell.setValue(3);
    expect(callback.values).toEqual([4]);
  });

  test('callbacks can be added and removed', () => {
    const inputCell = new InputCell(1);
    const output = new ComputeCell(
      [inputCell],
      (inputs) => inputs[0].value + 1,
    );

    const callback1 = new CallbackCell((cell) => cell.value);
    const callback2 = new CallbackCell((cell) => cell.value);

    output.addCallback(callback1);
    output.addCallback(callback2);

    inputCell.setValue(31);

    output.removeCallback(callback1);

    const callback3 = new CallbackCell((cell) => cell.value);
    output.addCallback(callback3);

    inputCell.setValue(41);

    expect(callback1.values).toEqual([32]);
    expect(callback2.values).toEqual([32, 42]);
    expect(callback3.values).toEqual([42]);
  });

  test("callbacks should not be called if dependencies change but output value doesn't change", () => {
    const inputCell = new InputCell(1);
    const plusOne = new ComputeCell(
      [inputCell],
      (inputs) => inputs[0].value + 1,
    );

    const minusOne = new ComputeCell(
      [inputCell],
      (inputs) => inputs[0].value - 1,
    );

    const alwaysTwo = new ComputeCell(
      [plusOne, minusOne],
      (inputs) => inputs[0].value - inputs[1].value,
    );

    const callback = new CallbackCell((cell) => cell.value);
    alwaysTwo.addCallback(callback);

    inputCell.setValue(2);
    inputCell.setValue(3);
    inputCell.setValue(4);
    inputCell.setValue(5);

    expect(callback.values).toEqual([]);
  });
});
