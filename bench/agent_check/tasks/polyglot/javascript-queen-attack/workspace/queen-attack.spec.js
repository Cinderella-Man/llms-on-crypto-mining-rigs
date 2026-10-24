import { describe, expect, test, xtest } from '@jest/globals';
import { QueenAttack } from './queen-attack';

describe('Queens', () => {
  describe('Test creation of Queens with valid and invalid positions', () => {
    test('queen with a valid position', () => {
      const queens = new QueenAttack({ white: [2, 2] });
      expect(queens.white).toEqual([2, 2]);
    });

    test('queen must have positive column', () => {
      const positioning = { white: [2, -2] };
      const expectedError = 'Queen must be placed on the board';
      expect(() => new QueenAttack(positioning)).toThrow(expectedError);
    });
  });

  describe('Test the ability of one queen to attack another', () => {
    test('cannot attack', () => {
      const queens = new QueenAttack({ white: [2, 4], black: [6, 6] });
      expect(queens.canAttack).toEqual(false);
    });

    test('can attack on first diagonal', () => {
      const queens = new QueenAttack({ white: [2, 2], black: [0, 4] });
      expect(queens.canAttack).toEqual(true);
    });

    test('can attack on fourth diagonal', () => {
      const queens = new QueenAttack({ white: [1, 7], black: [0, 6] });
      expect(queens.canAttack).toEqual(true);
    });

    test('cannot attack if falling diagonals are only the same when reflected across the longest falling diagonal', () => {
      const queens = new QueenAttack({ white: [4, 1], black: [2, 5] });
      expect(queens.canAttack).toEqual(false);
    });
  });

  describe('Test the board visualisation', () => {
    test('board', () => {
      const positioning = { white: [3, 2], black: [6, 5] };
      const queens = new QueenAttack(positioning);
      const board = [
        '_ _ _ _ _ _ _ _',
        '_ _ _ _ _ _ _ _',
        '_ _ _ _ _ _ _ _',
        '_ _ W _ _ _ _ _',
        '_ _ _ _ _ _ _ _',
        '_ _ _ _ _ _ _ _',
        '_ _ _ _ _ B _ _',
        '_ _ _ _ _ _ _ _',
      ].join('\n');
      expect(queens.toString()).toEqual(board);
    });

    test('board with queens at the edges', () => {
      const positioning = { white: [0, 0], black: [7, 7] };
      const queens = new QueenAttack(positioning);
      const board = [
        'W _ _ _ _ _ _ _',
        '_ _ _ _ _ _ _ _',
        '_ _ _ _ _ _ _ _',
        '_ _ _ _ _ _ _ _',
        '_ _ _ _ _ _ _ _',
        '_ _ _ _ _ _ _ _',
        '_ _ _ _ _ _ _ _',
        '_ _ _ _ _ _ _ B',
      ].join('\n');
      expect(queens.toString()).toEqual(board);
    });
  });
});
