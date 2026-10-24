# These tests are auto-generated with test data from:
# https://github.com/exercism/problem-specifications/tree/main/exercises/connect/canonical-data.json
# File last updated on 2026-05-01

import unittest

from connect import (
    ConnectGame,
)


class ConnectTest(unittest.TestCase):

    def test_an_empty_board_has_no_winner(self):
        game = ConnectGame(
            """. . . . .
                . . . . .
                 . . . . .
                  . . . . .
                   . . . . ."""
        )
        winner = game.get_winner()
        self.assertEqual(winner, "")

    def test_only_edges_does_not_make_a_winner(self):
        game = ConnectGame(
            """O O O X
                X . . X
                 X . . X
                  X O O O"""
        )
        winner = game.get_winner()
        self.assertEqual(winner, "")

    def test_x_wins_crossing_from_left_to_right(self):
        game = ConnectGame(
            """. O . .
                O X X X
                 O X O .
                  X X O X
                   . O X ."""
        )
        winner = game.get_winner()
        self.assertEqual(winner, "X")

    def test_o_wins_crossing_from_top_to_bottom(self):
        game = ConnectGame(
            """. O . .
                O X X X
                 O O O .
                  X X O X
                   . O X ."""
        )
        winner = game.get_winner()
        self.assertEqual(winner, "O")
