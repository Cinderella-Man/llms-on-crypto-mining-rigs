# These tests are auto-generated with test data from:
# https://github.com/exercism/problem-specifications/tree/main/exercises/go-counting/canonical-data.json
# File last updated on 2026-02-19

import unittest

from go_counting import (
    Board,
    WHITE,
    BLACK,
    NONE,
)


class GoCountingTest(unittest.TestCase):

    def test_black_corner_territory_on_5x5_board(self):
        board = Board(["  B  ", " B B ", "B W B", " W W ", "  W  "])
        stone, territory = board.territory(x=0, y=1)
        self.assertEqual(stone, BLACK)
        self.assertSetEqual(territory, {(0, 0), (0, 1), (1, 0)})

    def test_a_stone_and_not_a_territory_on_5x5_board(self):
        board = Board(["  B  ", " B B ", "B W B", " W W ", "  W  "])
        stone, territory = board.territory(x=1, y=1)
        self.assertEqual(stone, NONE)
        self.assertSetEqual(territory, set())

    def test_invalid_because_y_is_too_low_for_5x5_board(self):
        board = Board(["  B  ", " B B ", "B W B", " W W ", "  W  "])
        with self.assertRaises(ValueError) as err:
            board.territory(x=1, y=-1)
        self.assertEqual(type(err.exception), ValueError)
        self.assertEqual(err.exception.args[0], "Invalid coordinate")

    def test_two_territory_rectangular_board(self):
        board = Board([" BW ", " BW "])
        territories = board.territories()
        self.assertSetEqual(territories[BLACK], {(0, 0), (0, 1)})
        self.assertSetEqual(territories[WHITE], {(3, 0), (3, 1)})
        self.assertSetEqual(territories[NONE], set())
