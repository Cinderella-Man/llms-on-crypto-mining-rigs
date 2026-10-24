import { describe, expect, test, xtest } from '@jest/globals';
import { ResistorColorTrio } from './resistor-color-trio';

function makeLabel({ value, unit }) {
  return `Resistor value: ${value} ${unit}`;
}

describe('Resistor Color Trio', () => {
  test('Orange and orange and black', () => {
    expect(new ResistorColorTrio(['orange', 'orange', 'black']).label).toEqual(
      makeLabel({ value: 33, unit: 'ohms' }),
    );
  });

  test('Green and brown and orange', () => {
    expect(new ResistorColorTrio(['green', 'brown', 'orange']).label).toEqual(
      makeLabel({ value: 51, unit: 'kiloohms' }),
    );
  });

  test('Minimum possible value', () => {
    expect(new ResistorColorTrio(['black', 'black', 'black']).label).toEqual(
      makeLabel({ value: 0, unit: 'ohms' }),
    );
  });

  test('Ignore extra colors', () => {
    expect(
      new ResistorColorTrio(['blue', 'green', 'yellow', 'orange']).label,
    ).toEqual(makeLabel({ value: 650, unit: 'kiloohms' }));
  });
});
