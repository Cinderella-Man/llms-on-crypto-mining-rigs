import { describe, expect, test, xtest } from '@jest/globals';
import { Bowling } from './bowling';

describe('Bowling', () => {
  describe('Check game can be scored correctly.', () => {
    test('should be able to score a game with all zeros', () => {
      const rolls = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      ];
      const bowling = new Bowling();
      rolls.forEach((roll) => {
        bowling.roll(roll);
      });
      expect(bowling.score()).toEqual(0);
    });

    test('points scored in the roll after a spare are counted twice', () => {
      const rolls = [
        6, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      ];
      const bowling = new Bowling();
      rolls.forEach((roll) => {
        bowling.roll(roll);
      });
      expect(bowling.score()).toEqual(16);
    });

    test('a strike earns ten points in a frame with a single roll', () => {
      const rolls = [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
      const bowling = new Bowling();
      rolls.forEach((roll) => {
        bowling.roll(roll);
      });
      expect(bowling.score()).toEqual(10);
    });

    test('a strike in the last frame gets a two roll bonuses that is counted once', () => {
      const rolls = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 7, 1,
      ];
      const bowling = new Bowling();
      rolls.forEach((roll) => {
        bowling.roll(roll);
      });
      expect(bowling.score()).toEqual(18);
    });

    test('last two strikes followed by only last bonus with non strike points', () => {
      const rolls = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 0, 1,
      ];
      const bowling = new Bowling();
      rolls.forEach((roll) => {
        bowling.roll(roll);
      });
      expect(bowling.score()).toEqual(31);
    });
  });

  describe('Check game rules.', () => {
    test('rolls cannot score negative points', () => {
      const bowling = new Bowling();
      expect(() => {
        bowling.roll(-1);
      }).toThrow(new Error('Negative roll is invalid'));
    });

    test('bonus roll after a strike in the last frame cannot score more than 10 points', () => {
      const rolls = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10];
      const bowling = new Bowling();
      rolls.forEach((roll) => {
        bowling.roll(roll);
      });
      expect(() => {
        bowling.roll(11);
      }).toThrow(new Error('Pin count exceeds pins on the lane'));
    });

    test('the second bonus rolls after a strike in the last frame cannot be a strike if the first one is not a strike', () => {
      const rolls = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 6,
      ];
      const bowling = new Bowling();
      rolls.forEach((roll) => {
        bowling.roll(roll);
      });
      expect(() => {
        bowling.roll(10);
      }).toThrow(new Error('Pin count exceeds pins on the lane'));
    });

    test('an incomplete game cannot be scored', () => {
      const rolls = [0, 0];
      const bowling = new Bowling();
      rolls.forEach((roll) => {
        bowling.roll(roll);
      });
      expect(() => {
        bowling.score();
      }).toThrow(new Error('Score cannot be taken until the end of the game'));
    });

    test('both bonus rolls for a strike in the last frame must be rolled before score can be calculated', () => {
      const rolls = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10,
      ];
      const bowling = new Bowling();
      rolls.forEach((roll) => {
        bowling.roll(roll);
      });
      expect(() => {
        bowling.score();
      }).toThrow(new Error('Score cannot be taken until the end of the game'));
    });

    test('cannot roll after bonus rolls for strike', () => {
      const rolls = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 3, 2,
      ];
      const bowling = new Bowling();
      rolls.forEach((roll) => {
        bowling.roll(roll);
      });
      expect(() => {
        bowling.roll(2);
      }).toThrow(new Error('Cannot roll after game is over'));
    });
  });
});
