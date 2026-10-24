import { describe, expect, test, xtest } from '@jest/globals';
import { tournamentTally } from './tournament';

describe('Tournament', () => {
  test('just the header if no input', () => {
    const tally = tournamentTally('');
    const expected = 'Team                           | MP |  W |  D |  L |  P';
    expect(tally).toEqual(expected);
  });
  test('a different team can win', () => {
    const tally = tournamentTally('Blithering Badgers;Allegoric Alaskans;win');
    const expected =
      'Team                           | MP |  W |  D |  L |  P\n' +
      'Blithering Badgers             |  1 |  1 |  0 |  0 |  3\n' +
      'Allegoric Alaskans             |  1 |  0 |  0 |  1 |  0';
    expect(tally).toEqual(expected);
  });
  test('there can be more than one winner', () => {
    const input =
      'Allegoric Alaskans;Blithering Badgers;loss\n' +
      'Allegoric Alaskans;Blithering Badgers;win';
    const tally = tournamentTally(input);
    const expected =
      'Team                           | MP |  W |  D |  L |  P\n' +
      'Allegoric Alaskans             |  2 |  1 |  0 |  1 |  3\n' +
      'Blithering Badgers             |  2 |  1 |  0 |  1 |  3';
    expect(tally).toEqual(expected);
  });
  test('incomplete competition (not all pairs have played)', () => {
    const input =
      'Allegoric Alaskans;Blithering Badgers;loss\n' +
      'Devastating Donkeys;Allegoric Alaskans;loss\n' +
      'Courageous Californians;Blithering Badgers;draw\n' +
      'Allegoric Alaskans;Courageous Californians;win';
    const tally = tournamentTally(input);
    const expected =
      'Team                           | MP |  W |  D |  L |  P\n' +
      'Allegoric Alaskans             |  3 |  2 |  0 |  1 |  6\n' +
      'Blithering Badgers             |  2 |  1 |  1 |  0 |  4\n' +
      'Courageous Californians        |  2 |  0 |  1 |  1 |  1\n' +
      'Devastating Donkeys            |  1 |  0 |  0 |  1 |  0';
    expect(tally).toEqual(expected);
  });
});
