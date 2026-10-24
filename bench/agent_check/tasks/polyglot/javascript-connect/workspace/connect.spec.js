import { describe, expect, test, xtest } from '@jest/globals';
import { Board } from './connect';

describe('Judging a game of connect', () => {
  test('an empty board has no winner', () => {
    const board = [
      '. . . . .',
      ' . . . . .',
      '  . . . . .',
      '   . . . . .',
      '    . . . . .',
    ];
    expect(new Board(board).winner()).toEqual('');
  });

  test('only edges does not make a winner', () => {
    const board = ['O O O X', ' X . . X', '  X . . X', '   X O O O'];
    expect(new Board(board).winner()).toEqual('');
  });

  test('X wins crossing from left to right', () => {
    const board = [
      '. O . .',
      ' O X X X',
      '  O X O .',
      '   X X O X',
      '    . O X .',
    ];
    expect(new Board(board).winner()).toEqual('X');
  });

  test('O wins crossing from top to bottom', () => {
    const board = [
      '. O . .',
      ' O X X X',
      '  O O O .',
      '   X X O X',
      '    . O X .',
    ];
    expect(new Board(board).winner()).toEqual('O');
  });
});
